#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/vignesh_util.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void IdToWeightMappingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const int num_output = this->layer_param_.id_to_weight_mapping_param().num_output();
  
  K_ = this->layer_param_.id_to_weight_mapping_param().max_ids();
  N_ = num_output;
  CHECK_EQ(bottom[0]->count(), bottom[0]->num());

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {

    this->blobs_.resize(1);
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(K_, N_, 1, 1));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.id_to_weight_mapping_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  float norm = 0, maxval = -1;
  const Dtype* blob_data = this->blobs_[0]->cpu_data();
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 64; ++j) {
      norm += static_cast<float>(blob_data[i*64 + j]) * static_cast<float>(blob_data[i*64 + j]);
      if (static_cast<float>(blob_data[i*64 + j]) > maxval)
        maxval = static_cast<float>(blob_data[i*64 + j]);
    }
  }
  LOG(INFO) << "Norm of init 10 vectors: " << (norm/640) << " , maxval: " << maxval;


}

template <typename Dtype>
void IdToWeightMappingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Figure out the dimensions
  M_ = bottom[0]->num();
  CHECK_EQ(bottom[0]->count(), M_) << "Input size "
    "incompatible with inner product parameters.";
  (*top)[0]->Reshape(M_, N_, 1, 1);
}

template <typename Dtype>
void IdToWeightMappingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  //LOG(INFO) << "Trying to set the weight vector ... ";
  //caffe_copy_indices<Dtype>(M_, N_, weight, bottom_data, top_data);
  
  for (int i = 0; i < M_; ++i) {
    //LOG(INFO) << "Forwarding: " << static_cast<int>(bottom_data[i]);
    caffe_copy(N_,
        weight + this->blobs_[0]->offset(static_cast<int>(bottom_data[i]), 0, 0, 0),
        top_data + (*top)[0]->offset(i, 0, 0, 0));
  }

  /* (Debugging lines. TODO: Remove later)
  string wv = "";
  string indices = "";
  string wv1 = "";
  int top_index = static_cast<int>(bottom_data[0]);
  for (int i = 0; i < N_; ++i) {
    wv += stringprintf(":%f", static_cast<float>(top_data[i]));
    indices += stringprintf(":%d", static_cast<long>(bottom_data[i]));
    wv1 += stringprintf(":%f", static_cast<float>(weight[top_index*N_ + i]));
  }
  LOG(INFO) << "Weight: " << wv;
  LOG(INFO) << "Weight orig: " << wv1;
  LOG(INFO) << "Indices: " << indices;*/

}

template <typename Dtype>
void IdToWeightMappingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {

  if (propagate_down[0]) {
    LOG(FATAL) << this->type_name()
               << "Layer cannot backpropogate to input ids.";
  }

  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    Dtype* bottom_diff = this->blobs_[0]->mutable_cpu_diff();
    caffe_set<Dtype>(K_*N_, Dtype(0), bottom_diff);

    for (int i = 0; i < M_; ++i) {
      //LOG(INFO) << "Updating: " << static_cast<int>(bottom_data[i]);
      caffe_axpy<Dtype>(N_, 1, top_diff + top[0]->offset(i, 0, 0, 0),
          bottom_diff + this->blobs_[0]->offset(static_cast<int>(bottom_data[i]), 0, 0, 0));
    }

    //caffe_axpy_indices<Dtype>(M_, N_, top_diff, bottom_data, this->blobs_[0]->mutable_cpu_diff());


    /*double diff_norm = 0;
    double blob_norm = 0;
    for ( int i = 0; i < K_; ++i) {
      for (int j = 0; j < N_; ++j) {
        diff_norm += this->blobs_[0]->diff_at(i,j,0,0)*this->blobs_[0]->diff_at(i,j,0,0);
        blob_norm += this->blobs_[0]->data_at(i,j,0,0)*this->blobs_[0]->data_at(i,j,0,0);
      }
    }

    LOG(INFO) << "Total diff norm: " << diff_norm << ", blob-norm: " << blob_norm;*/

    /*
    const Dtype* blob_data = this->blobs_[0]->cpu_data();
    float norm = 0, top_diff_norm = 0, maxval = -1, maxdiff = -1;
    for (int i = 0; i < 10; ++i) {
      int id = static_cast<int>(bottom_data[i]);
      for (int j = 0; j < 64; ++j) {
        top_diff_norm += static_cast<float>(top_diff[i*64 + j]) * static_cast<float>(top_diff[i*64 + j]);
        norm += static_cast<float>(blob_data[id*64 + j]) * static_cast<float>(blob_data[id*64 + j]);
        if (static_cast<float>(blob_data[id*64 + j]) > maxval)
          maxval = static_cast<float>(blob_data[id*64 + j]);
        if (static_cast<float>(top_diff[i*64 + j]) > maxdiff)
          maxdiff = static_cast<float>(top_diff[i*64 + j]);

      }
    }
    LOG(INFO) << "Norm of first 100 vectors: " << (norm/640) << " , top-diff-norm: " << (top_diff_norm/640)
      << " , maxval: " << maxval << " , maxdiff: " << maxdiff;
    */
  }
}

#ifdef CPU_ONLY
STUB_GPU(IdToWeightMappingLayer);
#endif

INSTANTIATE_CLASS(IdToWeightMappingLayer);

}  // namespace caffe
