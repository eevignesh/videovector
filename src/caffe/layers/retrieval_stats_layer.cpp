#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/vignesh_util.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


DEFINE_int32(num_classes, 15, "Number of classes");

namespace caffe {

template <typename Dtype>
void RetrievalStatsLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {

  string id_to_class_file = this->layer_param_.retrieval_stats_param().id_to_class_file();

  stats_output_file_ = this->layer_param_.retrieval_stats_param().stats_output_file();
  exclude_same_video_shots_ = this->layer_param_.retrieval_stats_param().exclude_same_video_shots();

  // Read the video-id to class map from the file
  ifstream id2class_stream;
  id2class_stream.open(id_to_class_file.c_str());
  string line;
  while (getline(id2class_stream, line)) {
    vector<string> line_splits = strsplit(line, ",");
    CHECK_EQ(2, line_splits.size());
    std::string::size_type sval;
    int video_id = stoi(line_splits[0], &sval);
    CHECK_EQ(sval, line_splits[0].size());
    int class_id = stoi(line_splits[1], &sval);
    CHECK_EQ(sval, line_splits[1].size());
    video_id_to_class_.insert(make_pair(video_id, class_id));
  }
  id2class_stream.close();

  batch_size_ = bottom[0]->num();
  feature_dimension_ = bottom[0]->count()/bottom[0]->num();

  CHECK_GE(video_id_to_class_.size(), 1) << "need atleast one entry in id-to-class map!";
  

  if (this->layer_param_.retrieval_stats_param().video_level_retrieval()) {
    max_num_videos_ = this->layer_param_.retrieval_stats_param().max_num_videos();
    CHECK_GE(max_num_videos_, 1) << "To do video level retrieval ... need min 1 video";
    temp_matrix_.Reshape(max_num_videos_, batch_size_, 1, 1);
    temp_matrix_2_.Reshape(max_num_videos_, feature_dimension_, 1, 1);
    temp_video_ids_.Reshape(max_num_videos_, 1, 1, 1);
    caffe_set(temp_matrix_.count(), Dtype(0), temp_matrix_.mutable_cpu_data());

    distance_matrix_.Reshape(max_num_videos_, max_num_videos_, 1, 1);
    //norm_matrix_.Reshape(max_num_videos_, 1, 1, 1);
    //sum_multiplier_n_.Reshape(max_num_videos_, 1, 1, 1);
    //caffe_set(sum_multiplier_n_.count(), Dtype(1), sum_multiplier_n_.mutable_cpu_data());

  } else {
  
    distance_matrix_.Reshape(batch_size_, batch_size_, 1, 1);
    //norm_matrix_.Reshape(batch_size_, 1, 1, 1);
    //sum_multiplier_n_.Reshape(batch_size_, 1, 1, 1);
    //caffe_set(sum_multiplier_n_.count(), Dtype(1), sum_multiplier_n_.mutable_cpu_data());

  }

}

template <typename Dtype>
void RetrievalStatsLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);

  (*top)[0]->Reshape(1, 1, 1, 1); // Mean ap 
  (*top)[1]->Reshape(1, 1, 1, 1); // hit@1
  (*top)[2]->Reshape(1, 1, 1, 1); // hit@5
}

template <typename Dtype>
struct SortByDistance {

  bool operator() (int i, int j) { 
    return (distance_[i] < distance_[j]);
  }

  SortByDistance(const Dtype* data) {
    distance_ = data;
  }

  const Dtype* distance_;

};

template <typename Dtype>
void RetrievalStatsLayer<Dtype>:: ComputeStats(const Dtype* video_ids, const vector<int>& sort_ids,
    double& ap, double& acc_1,
    double& acc_5, const int current_video_id) {
  ap = 0; acc_1 = 0; acc_5 = 0;
  double val = 0, ret = 0;
  int current_class_id = video_id_to_class_[current_video_id];
  //vector<int> precisions;
  // Note, the first shot is always excluded ... since it is the same shot
  for (int i = 1; i < sort_ids.size(); ++i) {
    if ( (static_cast<int>(video_ids[sort_ids[i]]) != current_video_id) ||
        !this->exclude_same_video_shots_) {
      val++;
      if (video_id_to_class_[static_cast<int>(video_ids[sort_ids[i]])] == current_class_id) {
        if (val <= 1) {
          acc_1++;
        }
        if (val <= 5) {
          acc_5++;
        }
        ret++;
        ap += ret/val;
        //precisions.push_back(val);
      }
    }
  }

  if (ret > 0) {
    ap /= ret;
  }

  acc_5 /= 5;
  /*string precision_string = "";
  for (int i = 0; i < precisions.size(); ++i) {
    precision_string += stringprintf("%d:", precisions[i]);
  }
  LOG(INFO) << "Precision-string:" << precision_string;*/
}

template <typename Dtype>
void RetrievalStatsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  ofstream stats_output, top5_output;

  if (stats_output_file_ != "") {
    stats_output.open(stats_output_file_);
    stats_output << "#video_id,class_id,ap,acc@1,acc@5" 
                 << ",ret_id_1,ret_id_2,ret_id_3,ret_id_4,ret_id_5"
                 << ",class_id_1,class_id_2,class_id_3,class_id_4,class_id_5" << std::endl;
  }

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_video_ids = bottom[1]->cpu_data();
  //Dtype* norm_data = norm_matrix_.mutable_cpu_data();
  int num_samples = 0;
  // Vector norms
  //caffe_powx(bottom[0]->count(), bottom_data, Dtype(2), temp_matrix_.mutable_cpu_data());
  //caffe_cpu_gemv<Dtype>(CblasNoTrans, batch_size_, feature_dimension_, 1,
  //    temp_matrix_.cpu_data(), sum_multiplier_f_.cpu_data(), 0., norm_matrix_.mutable_cpu_data());

  if (this->layer_param_.retrieval_stats_param().video_level_retrieval()) {

    Dtype* temp_mutable_data = temp_matrix_.mutable_cpu_data();
    boost::unordered_map<int, int> num_shots_per_video;
    boost::unordered_map<int, int> new_to_old_video_id_map;

    for (int i = 0; i < batch_size_; ++i) {
      if (num_shots_per_video.find(bottom_video_ids[i]) == num_shots_per_video.end()) {
        num_shots_per_video[bottom_video_ids[i]] = 1;
      } else {
        num_shots_per_video[bottom_video_ids[i]] += 1;
      }
    }

    int it_ctr = 0;
    Dtype* temp_video_ids_data = temp_video_ids_.mutable_cpu_data();
    for (auto map_it = num_shots_per_video.begin(); map_it != num_shots_per_video.end(); ++map_it) {
      new_to_old_video_id_map[(*map_it).first] = it_ctr;
      *(temp_video_ids_data + temp_video_ids_.offset(it_ctr, 0, 0, 0)) = (*map_it).first;
      it_ctr++;
    }

    CHECK_EQ(num_shots_per_video.size(), max_num_videos_);

    for (int i = 0; i < batch_size_; ++i) {
      CHECK_GE(num_shots_per_video[bottom_video_ids[i]], 1.0);
      Dtype* temp_val = temp_mutable_data + temp_matrix_.offset(static_cast<int>
        (new_to_old_video_id_map[bottom_video_ids[i]]),i,0,0);
      temp_val[0] = 1.0/num_shots_per_video[bottom_video_ids[i]];

    }

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, max_num_videos_, feature_dimension_, batch_size_,
        (Dtype)(1.0), temp_matrix_.cpu_data(), bottom_data, (Dtype)0., temp_matrix_2_.mutable_cpu_data());

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, max_num_videos_,
        max_num_videos_, feature_dimension_,
        (Dtype)(-2.0), temp_matrix_2_.cpu_data(), temp_matrix_2_.cpu_data(),
        (Dtype)0., distance_matrix_.mutable_cpu_data());

    num_samples = max_num_videos_;
  } else {

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, batch_size_, batch_size_, feature_dimension_,
        (Dtype)(-2.0), bottom_data, bottom_data, (Dtype)0., distance_matrix_.mutable_cpu_data());
    num_samples = batch_size_;
  }

  double mean_ap = 0, mean_acc_1 = 0, mean_acc_5 = 0;
  double num_positives = 0;
  std::vector<int> sort_ids(num_samples);
  vector<int> top_5_ids(5,0);

  vector<double> inter_class_distance(FLAGS_num_classes, 0);
  vector<double> intra_class_distance(FLAGS_num_classes, 0);
  vector<double> num_inter_class(FLAGS_num_classes, 1);
  vector<double> num_intra_class(FLAGS_num_classes, 1);
  vector<double> ap_per_class(FLAGS_num_classes, 0);
  vector<double> num_samples_per_class(FLAGS_num_classes, 0);
  double dist_value = 0;

  for (int i = 0; i < num_samples; ++i) {
    double ap, acc_1, acc_5;
    std::iota(sort_ids.begin(), sort_ids.end(), 0);


    Dtype* same_val = distance_matrix_.mutable_cpu_data() + distance_matrix_.offset(i,i,0,0);
    same_val[0] = -1e15; // set the diagonal to a very small value

    SortByDistance<Dtype> sbd(distance_matrix_.cpu_data() +
        distance_matrix_.offset(i, 0, 0, 0));
    std::sort(sort_ids.begin(), sort_ids.end(), sbd);
    
    CHECK_EQ(sort_ids[0], i);
    
    int sample_label = -100;


    // -------------------------- Don't compute mAP for samples with labels < 0 -----------------------
    if (!this->layer_param_.retrieval_stats_param().video_level_retrieval()) {
      sample_label = video_id_to_class_[static_cast<int>(bottom_video_ids[i])]; 
    } else {
      sample_label = video_id_to_class_[static_cast<int>(temp_video_ids_.cpu_data()[i])]; 
    }

    if (sample_label < 0) {
      continue;
    }


    // TODO: Remove later ... only for debugging now
    /*if (!this->layer_param_.retrieval_stats_param().video_level_retrieval()) {
      for (int samp = 0; samp < num_samples; ++samp)  {
        if (samp == i) {
          continue;
        }
        dist_value = static_cast<double>(*(distance_matrix_.cpu_data() + distance_matrix_.offset(i,samp,0,0)));
        if (video_id_to_class_[static_cast<int>(bottom_video_ids[samp])] == sample_label) {
          inter_class_distance[sample_label-1] += dist_value;
          num_inter_class[sample_label-1]++;
        } else {
          intra_class_distance[sample_label-1] += dist_value;
          num_intra_class[sample_label-1]++;
        }

      }
    } else {
      for (int samp = 0; samp < num_samples; ++samp) {
        if (samp == i) {
          continue;
        }
        dist_value = static_cast<double>(*(distance_matrix_.cpu_data() + distance_matrix_.offset(i,samp,0,0)));
         if (video_id_to_class_[static_cast<int>(temp_video_ids_.cpu_data()[samp])] == sample_label) {
            inter_class_distance[sample_label-1] += dist_value;
            num_inter_class[sample_label-1]++;
          } else {
            intra_class_distance[sample_label-1] += dist_value;
            num_intra_class[sample_label-1]++;
          }
      }  
    }*/

    // -------------------------- Don't compute mAP for samples with labels < 0 -----------------------

    if (!this->layer_param_.retrieval_stats_param().video_level_retrieval()) {
      ComputeStats(bottom_video_ids, sort_ids, ap, acc_1,
          acc_5, static_cast<int>(bottom_video_ids[i]));
    } else {
      ComputeStats(temp_video_ids_.cpu_data(), sort_ids, ap, acc_1,
          acc_5, static_cast<int>(temp_video_ids_.cpu_data()[i]));
    }

    if (sample_label >= 0) {
      mean_ap += ap;
      mean_acc_1 += acc_1;
      mean_acc_5 += acc_5;
      num_positives++;
      //ap_per_class[sample_label-1] += ap;
      //num_samples_per_class[sample_label-1]++;
    }

    if (stats_output_file_ != "" && sample_label >= 0) {
      if (!this->layer_param_.retrieval_stats_param().video_level_retrieval()) {

        // Get top-5 ids
        int ctr_j = 0;
        for (int jj = 0; (jj < num_samples) && (ctr_j < 5) ; ++jj) {
          if (bottom_video_ids[sort_ids[jj]] != bottom_video_ids[i]) {
            top_5_ids[ctr_j] = sort_ids[jj];
            ctr_j++;
          } 
        }

        stats_output << static_cast<int>(bottom_video_ids[i]) << ","
          << video_id_to_class_[static_cast<int>(bottom_video_ids[i])]
          << "," << ap << "," << acc_1 << "," << acc_5 << ","
          << top_5_ids[0] << ","
          << top_5_ids[1] << ","
          << top_5_ids[2] << ","
          << top_5_ids[3] << ","
          << top_5_ids[4] << ","
          << video_id_to_class_[static_cast<int>(bottom_video_ids[top_5_ids[0]])] << ","
          << video_id_to_class_[static_cast<int>(bottom_video_ids[top_5_ids[1]])] << ","
          << video_id_to_class_[static_cast<int>(bottom_video_ids[top_5_ids[2]])] << ","
          << video_id_to_class_[static_cast<int>(bottom_video_ids[top_5_ids[3]])] << ","
          << video_id_to_class_[static_cast<int>(bottom_video_ids[top_5_ids[4]])]
          << std::endl;
      } else {
        stats_output << static_cast<int>(temp_video_ids_.cpu_data()[i]) << ","
          << video_id_to_class_[static_cast<int>(temp_video_ids_.cpu_data()[i])]
          << "," << ap << "," << acc_1 << "," << acc_5
          << std::endl;
      }
    }
  }

  /*for (int cid = 0; cid < FLAGS_num_classes; ++cid) {
    LOG(INFO) << "class: " << cid+1 << ":inter:" << inter_class_distance[cid]/num_inter_class[cid];
    LOG(INFO) << "class: " << cid+1 << ":intra:" << intra_class_distance[cid]/num_intra_class[cid];
    LOG(INFO) << "class: " << cid+1 << ":AP:" << ap_per_class[cid]/num_samples_per_class[cid];
  }*/

  if (stats_output_file_ != "") {
    stats_output.close();
  }

  (*top)[0]->mutable_cpu_data()[0] = mean_ap/num_positives;
  (*top)[1]->mutable_cpu_data()[0] = mean_acc_1/num_positives;
  (*top)[2]->mutable_cpu_data()[0] = mean_acc_5/num_positives;
  // This layer should not be used as a loss function.
}

INSTANTIATE_CLASS(RetrievalStatsLayer);

}  // namespace caffe
