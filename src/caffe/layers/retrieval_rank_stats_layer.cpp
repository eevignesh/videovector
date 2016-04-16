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

namespace caffe {

template <typename Dtype>
void RetrievalRankStatsLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {

  stats_output_file_ = this->layer_param_.retrieval_rank_stats_param().stats_output_file();
  
  //TODO: IMPLEMENT LATER
  exclude_same_video_shots_ = this->layer_param_.retrieval_rank_stats_param().exclude_same_video_shots();


  batch_size_ = bottom[0]->num();
  num_frames_ = bottom[1]->num();

  if (!this->layer_param_.retrieval_rank_stats_param().compute_ap()) {
    CHECK_EQ(batch_size_, num_frames_);
  }

  feature_dimension_ = bottom[0]->count()/bottom[0]->num();

  positive_size_ = this->layer_param_.retrieval_rank_stats_param().positive_size();
  negative_size_ = this->layer_param_.retrieval_rank_stats_param().negative_size();

  if (positive_size_ > 0) {
    num_videos_ = num_frames_ / (positive_size_ + negative_size_);
    CHECK_EQ(num_videos_, batch_size_);
  } else {
    num_videos_ = 0;
  }

  //CHECK_EQ(batch_size_, bottom[1]->num());
  CHECK_EQ(bottom[1]->count()/num_frames_, bottom[0]->count()/batch_size_);

  
  distance_matrix_.Reshape(batch_size_, num_frames_, 1, 1);

}

template <typename Dtype>
void RetrievalRankStatsLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  
  (*top)[0]->Reshape(1, 1, 1, 1); // Median rank 
  (*top)[1]->Reshape(1, 1, 1, 1); // recall@1
  (*top)[2]->Reshape(1, 1, 1, 1); // recall@5
  (*top)[3]->Reshape(1, 1, 1, 1); // recall@10
  (*top)[4]->Reshape(1, 1, 1, 1); // Mean AP

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
void RetrievalRankStatsLayer<Dtype>:: ComputeRankStats(const vector<int>& sort_ids,
    int& rank, double& rec_1, double& rec_5, double& rec_10,
    const int current_item_id) {


  // First set the rank
  auto item_index = std::find(sort_ids.begin(), sort_ids.end(), current_item_id);
  CHECK(item_index != sort_ids.end()) << "Could not find the retrieved id ... something wrong!!!";
  rank = std::distance(sort_ids.begin(), item_index) + 1;

  rec_1 = 0; rec_5 = 0; rec_10 = 0;

  if (rank == 1) {
    rec_1 = 1;
  }

  if (rank <= 5) {
    rec_5 = 1;
  }

  if (rank <= 10) {
    rec_10 = 1;
  }

}

template <typename Dtype>
int RetrievalRankStatsLayer<Dtype>::GetVideoId(int item_id) {

  int bucket_id = item_id / (num_videos_);
  // First positive_size buckets are positives and rest are negatives

  if (bucket_id >= positive_size_) {
    return -(item_id % num_videos_);
  } else {
    return item_id % (num_videos_);
  }

  
  /*int video_id = item_id / (negative_size_ + positive_size_);

  if (item_id % (negative_size_ + positive_size_) < positive_size_) {
    return video_id;
  } else {
    return -1;
  }*/

}

template <typename Dtype>
void RetrievalRankStatsLayer<Dtype>::ComputeApStats(const vector<int>& sort_ids,
    double& ap, double& acc_1,
    double& acc_5, double& acc_10,
    int& best_rank, const int current_video_id) {
  ap = 0; acc_1 = 0; acc_5 = 0; acc_10 = 0;
  double val = 0, ret = 0;
  best_rank = 10000;
  //vector<int> precisions;
  // Note, the first shot is always excluded ... since it is the same shot
  for (int i = 0; i < sort_ids.size(); ++i) {
    val++;
    if (GetVideoId(sort_ids[i]) == current_video_id) {

      if (val < best_rank) {
        best_rank = val;
      }

      if (val <= 1) {
        acc_1++;
      }
      if (val <= 5) {
        acc_5++;
      }
      if (val <= 10) {
        acc_10++;
      }
      ret++;
      ap += ret/val;
    }
  }

  if (ret > 0) {
    ap /= ret;

    if (ret < 5) {
      acc_5 /= ret;
    } else {
      acc_5 /= 5;
    }
    if (ret < 10) {
      acc_10 /= ret;
    } else {
      acc_10 /= 10;
    }
  }
  /*string precision_string = "";
  for (int i = 0; i < precisions.size(); ++i) {
    precision_string += stringprintf("%d:", precisions[i]);
  }
  LOG(INFO) << "Precision-string:" << precision_string;*/
}

template <typename Dtype>
void RetrievalRankStatsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  ofstream stats_output, top5_output;

  if (stats_output_file_ != "") {
    stats_output.open(stats_output_file_);
    stats_output << "#item_id,rank,rec@1,rec@5" 
                 << ",ret_id_1,ret_id_2,ret_id_3,ret_id_4,ret_id_5"
                 << std::endl;
  }

  const Dtype* bottom_target_data = bottom[1]->cpu_data();
  const Dtype* bottom_context_data = bottom[0]->cpu_data();
  int num_samples = 0;

  // Vector norms
  //caffe_powx(bottom[0]->count(), bottom_data, Dtype(2), temp_matrix_.mutable_cpu_data());
  //caffe_cpu_gemv<Dtype>(CblasNoTrans, batch_size_, feature_dimension_, 1,
  //    temp_matrix_.cpu_data(), sum_multiplier_f_.cpu_data(), 0., norm_matrix_.mutable_cpu_data());

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, batch_size_, num_frames_, feature_dimension_,
      (Dtype)(-2.0), bottom_context_data, bottom_target_data, (Dtype)0., distance_matrix_.mutable_cpu_data());



  

  num_samples = batch_size_;

  vector<int> all_ranks;
  double mean_recall_1 = 0, mean_recall_5 = 0, mean_recall_10 = 0, mean_ap = 0;
  std::vector<int> sort_ids(num_frames_);

  vector<int> top_5_ids(5,0);

  for (int i = 0; i < num_samples; ++i) {
    int rank;
    double rec_1, rec_5, rec_10, ap;
    std::iota(sort_ids.begin(), sort_ids.end(), 0);

    SortByDistance<Dtype> sbd(distance_matrix_.cpu_data() +
        distance_matrix_.offset(i, 0, 0, 0));
    std::sort(sort_ids.begin(), sort_ids.end(), sbd);
    
    string sort_id_string = "";
    for (int i = 0; i < 10; ++i) {
      sort_id_string += stringprintf("%d-%d:", sort_ids[i], GetVideoId(sort_ids[i]));
    }
    LOG(INFO) << "Id: " << i << " ==> " << sort_id_string;

    if (this->layer_param_.retrieval_rank_stats_param().compute_ap()) {
      ComputeApStats(sort_ids, ap, rec_1, rec_5, rec_10, rank, i);
      mean_ap += ap;
      all_ranks.push_back(rank);
    } else {
      ComputeRankStats(sort_ids, rank, rec_1, rec_5, rec_10, i);
      all_ranks.push_back(rank);
    }

    mean_recall_1 += rec_1;
    mean_recall_5 += rec_5;
    mean_recall_10 += rec_10;

    if (stats_output_file_ != "") {

      // Get top-5 ids
      for (int jj = 0; (jj < num_samples) && (jj < 5) ; ++jj) {
        top_5_ids[jj] = sort_ids[jj];
      }

      stats_output << i
        << "," << rank << "," << rec_1 << "," << rec_5 << ","
        << top_5_ids[0] << ","
        << top_5_ids[1] << ","
        << top_5_ids[2] << ","
        << top_5_ids[3] << ","
        << top_5_ids[4] << ","
        << float(*(distance_matrix_.cpu_data() + distance_matrix_.offset(i, top_5_ids[0], 0, 0))) << "," 
        << float(*(distance_matrix_.cpu_data() + distance_matrix_.offset(i, top_5_ids[1], 0, 0))) << ","
        << float(*(distance_matrix_.cpu_data() + distance_matrix_.offset(i, top_5_ids[2], 0, 0))) << ","
        << float(*(distance_matrix_.cpu_data() + distance_matrix_.offset(i, top_5_ids[3], 0, 0))) << ","
        << float(*(distance_matrix_.cpu_data() + distance_matrix_.offset(i, top_5_ids[4], 0, 0)))
        << std::endl;
    }

    /*if (i < 10) {
      LOG(INFO) << "Ap " << i << " : " << ap;
    }*/
  }

  for (int i = 0; i < 10; ++i) {
    string dist_string = "";
    for (int j = 0; j < 10; ++j) {
      dist_string += stringprintf("%.4f:", static_cast<double>(*(distance_matrix_.cpu_data() +
                                           distance_matrix_.offset(i,j,0,0))));
    }
    LOG(INFO) << "-----> " << i << " <----- : " << dist_string;
    if (i == (num_samples-1))
      break;
  }

  if (stats_output_file_ != "") {
    stats_output.close();
  }

  double median_rank = -1.0;
  std::sort(all_ranks.begin(), all_ranks.end());
  if (num_samples % 2 == 0) {
    median_rank = (double) (all_ranks[num_samples/2 -1] + all_ranks[num_samples/2] )/2.0;
  } else {
    median_rank = (double) (all_ranks[num_samples/2]);
  }

  (*top)[0]->mutable_cpu_data()[0] = Dtype(median_rank);
    
  if (this->layer_param_.retrieval_rank_stats_param().compute_ap()) {
    (*top)[4]->mutable_cpu_data()[0] = Dtype(mean_ap/num_samples);
  } else {
    (*top)[4]->mutable_cpu_data()[0] = Dtype(0);
  }

  (*top)[1]->mutable_cpu_data()[0] = Dtype(mean_recall_1/num_samples);
  (*top)[2]->mutable_cpu_data()[0] = Dtype(mean_recall_5/num_samples);
  (*top)[3]->mutable_cpu_data()[0] = Dtype(mean_recall_10/num_samples);

  // This layer should not be used as a loss function.
}

INSTANTIATE_CLASS(RetrievalRankStatsLayer);

}  // namespace caffe
