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
void RetrievalRankStatsFixedRefLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {

  stats_output_file_ = this->layer_param_.retrieval_rank_stats_fixed_ref_param().stats_output_file();
 
  batch_size_ = bottom[0]->num(); // Actual features
  CHECK_EQ(batch_size_, bottom[1]->num()); // video_ids
  feature_dimension_ = bottom[0]->count()/bottom[0]->num();
  num_reference_points_ = bottom[2]->num();
  CHECK_EQ(num_reference_points_, bottom[3]->num());

  // Reshape distance_matrix_
  distance_matrix_.Reshape(batch_size_, num_reference_points_, 1, 1);
  
}

template <typename Dtype>
void RetrievalRankStatsFixedRefLayer<Dtype>::Reshape(
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
void RetrievalRankStatsFixedRefLayer<Dtype>::ComputeApStats(const vector<int>& sort_ids,
    double& ap, double& acc_1,
    double& acc_5, double& acc_10,
    int& best_rank, const int current_video_id,
    const Dtype* reference_ids) {
  ap = 0; acc_1 = 0; acc_5 = 0; acc_10 = 0;
  double val = 0, ret = 0;
  best_rank = 10000;
  //vector<int> precisions;
  // Note, the first shot is always excluded ... since it is the same shot
  for (int i = 0; i < sort_ids.size(); ++i) {
    val++;
    if (static_cast<int>(*(reference_ids + sort_ids[i])) == current_video_id) {

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
void RetrievalRankStatsFixedRefLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  ofstream stats_output, top5_output;

  if (stats_output_file_ != "") {
    stats_output.open(stats_output_file_);
    stats_output << "#item_id,rank,rec@1,rec@5" 
                 << ",ret_id_1,ret_id_2,ret_id_3,ret_id_4,ret_id_5"
                 << std::endl;
  }

  const Dtype* bottom_video_ids = bottom[1]->cpu_data();
  const Dtype* bottom_context_data = bottom[0]->cpu_data();
  const Dtype* bottom_reference_ids = bottom[3]->cpu_data();
  const Dtype* bottom_reference_data = bottom[2]->cpu_data();

  int num_samples = 0;

  // Vector norms
  //caffe_powx(fixed_reference_features_->count(), fixed_reference_features_.cpu_data(), Dtype(2),
  //    fixed_reference_features_.mutable_cpu_data());
  //caffe_cpu_gemv<Dtype>(CblasNoTrans, batch_size_, feature_dimension_, 1,
  //    temp_matrix_.cpu_data(), sum_multiplier_f_.cpu_data(), 0., norm_matrix_.mutable_cpu_data());

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, batch_size_, num_reference_points_, feature_dimension_,
      (Dtype)(-2.0), bottom_context_data, bottom_reference_data,
      (Dtype)0., distance_matrix_.mutable_cpu_data());


  num_samples = batch_size_;

  vector<int> all_ranks;
  double mean_recall_1 = 0, mean_recall_5 = 0, mean_recall_10 = 0, mean_ap = 0;
  std::vector<int> sort_ids(num_reference_points_);

  vector<int> top_5_ids(5,0);

  for (int i = 0; i < num_samples; ++i) {
    int rank;
    double rec_1, rec_5, rec_10, ap;
    std::iota(sort_ids.begin(), sort_ids.end(), 0);

    SortByDistance<Dtype> sbd(distance_matrix_.cpu_data() +
        distance_matrix_.offset(i, 0, 0, 0));
    std::sort(sort_ids.begin(), sort_ids.end(), sbd);
    
    /*string sort_id_string = "";
    for (int i = 0; i < 10; ++i) {
      sort_id_string += stringprintf("%d(%d):", sort_ids[i], static_cast<int>(*(bottom_reference_ids + sort_ids[i])));
    }
    LOG(INFO) << "Id: " << static_cast<int>(*(bottom_video_ids + i)) << " ==> " << sort_id_string;
    */

    ComputeApStats(sort_ids, ap, rec_1, rec_5, rec_10, rank, static_cast<int>(*(bottom_video_ids + i)), bottom_reference_ids);
    mean_ap += ap;
    all_ranks.push_back(rank);
    mean_recall_1 += rec_1;
    mean_recall_5 += rec_5;
    mean_recall_10 += rec_10;

    if (stats_output_file_ != "") {

      // Get top-5 ids
      for (int jj = 0; (jj < num_samples) && (jj < 5) ; ++jj) {
        top_5_ids[jj] = sort_ids[jj];
      }

      stats_output << i << "," << static_cast<int>(*(bottom_video_ids + i))
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
  (*top)[4]->mutable_cpu_data()[0] = Dtype(mean_ap/num_samples);
  (*top)[1]->mutable_cpu_data()[0] = Dtype(mean_recall_1/num_samples);
  (*top)[2]->mutable_cpu_data()[0] = Dtype(mean_recall_5/num_samples);
  (*top)[3]->mutable_cpu_data()[0] = Dtype(mean_recall_10/num_samples);

  // This layer should not be used as a loss function.
}

INSTANTIATE_CLASS(RetrievalRankStatsFixedRefLayer);

}  // namespace caffe
