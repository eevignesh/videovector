#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void SocialPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  SocialPoolingParameter pool_param = this->layer_param_.social_pooling_param();
  pool_feat_size_ = pool_param.pool_feat_size();
  //pool_bin_size_ = pool_param.pool_bin_size();
  //
  this->blobs_.resize(2);
  this->blobs_[0].reset(new Blob<Dtype>(1, 1, 2, pool_feat_size_));
  this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, pool_feat_size_));

  shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
      this->layer_param_.inner_product_param().bias_filler()));
  bias_filler->Fill(this->blobs_[1].get());

  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
      this->layer_param_.inner_product_param().weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());

  this->param_propagate_down_.resize(2, true);
}

template <typename Dtype>
void SocialPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  B_ = bottom[0]->channels();
  CHECK_EQ(1, bottom[0]->num());
  feat_dim_ = bottom[0]->count() / bottom[0]->channels();
  CHECK_GE(feat_dim_, 2); 

  CHECK_EQ(B_, bottom[1]->channels());
  CHECK_EQ(B_, bottom[1]->count()/B_);

  CHECK_EQ(top->size(), 1);
  (*top)[0]->Reshape(1, B_, pool_feat_size_, 1);

  diff_.Reshape(2, B_, B_, 1);
  x_sel_.Reshape(1, 1, feat_dim_, 1);
  y_sel_.Reshape(1, 1, feat_dim_, 1);
  const_multiplier_.Reshape(1, 1, 1, B_*B_);
  temp_B_.Reshape(1, 1, B_, 1);
  mean_added_data_.Reshape(1, B_, feat_dim_, 1);

  diff_feat_.Reshape(1, B_, B_, pool_feat_size_);
  diff_feat_sig_.Reshape(1, B_, B_, pool_feat_size_);

  eye_.Reshape(1, B_, B_, 1);
  caffe_set<Dtype>(B_*B_, (Dtype)0., eye_.mutable_cpu_data());
  for (int i = 0; i < B_; ++i) {
    *(eye_.mutable_cpu_data() + i*B_ + i) = (Dtype)1.0;
  }

  caffe_set<Dtype>(feat_dim_, (Dtype)0.0, x_sel_.mutable_cpu_data());
  caffe_set<Dtype>(feat_dim_, (Dtype)0.0, y_sel_.mutable_cpu_data());
  caffe_set<Dtype>(B_*B_, (Dtype)1.0, const_multiplier_.mutable_cpu_data());

  *(x_sel_.mutable_cpu_data()) = (Dtype) 1.0;
  *(y_sel_.mutable_cpu_data() + 1) = (Dtype) 1.0;


  /* Create a matrix of the form [1 0 0 ..; 0 1 0  ...] (2 x feat_dim_)*/
  f22_mat_.Reshape(1, 2, feat_dim_, 1);
  caffe_set<Dtype>(2*feat_dim_, (Dtype)0., f22_mat_.mutable_cpu_data());
  *(f22_mat_.mutable_cpu_data()) = (Dtype)1.;
  *(f22_mat_.mutable_cpu_data() + feat_dim_ + 1) = (Dtype)1.;

  /*f11_mat_.Reshape(1, 1, feat_dim_, 1);
  caffe_set<Dtype>(feat_dim_, (Dtype)0., f11_mat_.mutable_cpu_data());
  *(f11_mat_.mutable_cpu_data()) = (Dtype)1.;
  *(f11_mat_.mutable_cpu_data() + 1) = (Dtype)1.;*/
 
}

// Pool in a region around each trajectory position 
template <typename Dtype>
void SocialPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* mean_added_data = NULL;
  if (bottom.size() > 2) {
    mean_added_data = mean_added_data_.mutable_cpu_data();
    //caffe_add<Dtype>(B_*feat_dim_, bottom_data, bottom[2]->cpu_data(), mean_added_data);
    caffe_copy<Dtype>(B_*feat_dim_, bottom_data, mean_added_data);
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, feat_dim_, 2, (Dtype)1.,
        bottom[2]->cpu_data(), f22_mat_.cpu_data(), (Dtype)1., mean_added_data);
  }
  
  const Dtype* bottom_related = bottom[1]->cpu_data();

  Dtype* diff_data = diff_.mutable_cpu_data();
  const Dtype* x_selector = x_sel_.cpu_data();
  const Dtype* y_selector = y_sel_.cpu_data();

  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype* temp_B_data = temp_B_.mutable_cpu_data();

  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* bias = this->blobs_[1]->cpu_data();

  Dtype* diff_feat = diff_feat_.mutable_cpu_data();
  Dtype* diff_feat_sig = diff_feat_sig_.mutable_cpu_data();

  // First get the x-difference in diff_data
  //
  if (bottom.size() > 2) {
    caffe_cpu_gemv<Dtype>(CblasNoTrans, B_, feat_dim_, (Dtype)1., mean_added_data,
        x_selector, (Dtype)0.0, temp_B_data); 
  } else {
    caffe_cpu_gemv<Dtype>(CblasNoTrans, B_, feat_dim_, (Dtype)1., bottom_data,
        x_selector, (Dtype)0.0, temp_B_data);
  }

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, B_, 1, (Dtype)1.0,
      temp_B_data, const_multiplier_.cpu_data(), (Dtype)0.0, diff_data);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, B_, 1, (Dtype)(-1.0),
      const_multiplier_.cpu_data(), temp_B_data, (Dtype)1.0, diff_data);

  // First get the y-difference in diff_data + (B_*B_)
  if (bottom.size() > 2) {
    caffe_cpu_gemv<Dtype>(CblasNoTrans, B_, feat_dim_, (Dtype)1., mean_added_data,
        y_selector, (Dtype)0.0, temp_B_data);
  } else {
    caffe_cpu_gemv<Dtype>(CblasNoTrans, B_, feat_dim_, (Dtype)1., bottom_data,
        y_selector, (Dtype)0.0, temp_B_data);
  }

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, B_, 1, (Dtype)1.0,
      temp_B_data, const_multiplier_.cpu_data(), (Dtype)0.0, diff_data + B_*B_);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, B_, 1, (Dtype)(-1.0),
      const_multiplier_.cpu_data(), temp_B_data, (Dtype)1.0, diff_data + B_*B_);

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_*B_, pool_feat_size_, 1, (Dtype)1.0,
    diff_data, weight, (Dtype)0.0, diff_feat);

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_*B_, pool_feat_size_, 1, (Dtype)1.0,
    diff_data + B_*B_, weight + pool_feat_size_, (Dtype)1.0, diff_feat);

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_*B_, pool_feat_size_, 1, (Dtype)1.0,
    const_multiplier_.cpu_data(), bias, (Dtype)1.0, diff_feat);

  sigmoid_->Forward_cpu(B_*B_*pool_feat_size_, diff_feat, diff_feat_sig);
  // Find the bucket into which each of the values fall (Only max pooling for now)
  //caffe_set<Dtype>(B_*pool_feat_size_, (Dtype)0.0, top_data);
  
  for (int i = 0; i < B_; ++i) {
    caffe_cpu_gemv<Dtype>(CblasTrans, pool_feat_size_, B_, (Dtype)1.0,
        diff_feat_sig + (i*B_*pool_feat_size_), bottom_related + i*B_, 
        (Dtype)0.0, top_data + i*pool_feat_size_);
  }

  /*int half_pool_size = pool_size_/2;
  int index_x = 0, index_y = 0;
  Dtype half_bin_size = (Dtype)(pool_bin_size_/2.0); 
  for (int i = 0; i < B_; ++i) {
    for (int j = 0; j < B_; ++j) {
      if (bottom[1]->data_at(0, i, j, 0) > (Dtype)0.0) {
        index_x = floor((double)(diff_.data_at(0, i, j, 0) + half_bin_size)/ (2* half_bin_size));
        index_x = index_x + half_pool_size;
        index_y = floor((double)(diff_.data_at(1, i, j, 0) + half_bin_size)/ (2* half_bin_size));
        index_y = index_y + half_pool_size;
        if ((index_x) >= 0 && (index_x) < pool_size_
            &&(index_y) >= 0 && (index_y) < pool_size_) {
          top_data[i*pool_size_*pool_size_ + index_x*pool_size_ + index_y] += 1.0;
        }
      }
    }
  }*/

}

template <typename Dtype>
void SocialPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {

  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set((*bottom)[0]->count(), Dtype(0), bottom_diff);

  //const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  const Dtype* bottom_related = (*bottom)[1]->cpu_data();

  const Dtype* diff_data = diff_.cpu_data();
  //const Dtype* diff_feat = diff_feat_.cpu_data();
  const Dtype* diff_feat_sig = diff_feat_sig_.cpu_data();

  Dtype* diff_grad = diff_.mutable_cpu_diff();
  Dtype* diff_feat_sig_grad = diff_feat_sig_.mutable_cpu_diff();
  Dtype* diff_feat_grad = diff_feat_.mutable_cpu_diff();

  Dtype* temp_B_data = temp_B_.mutable_cpu_data();

  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();

  const Dtype* weight = this->blobs_[0]->cpu_data();
  //const Dtype* bias = this->blobs_[1]->cpu_data();

  for (int i = 0; i < B_; ++i) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, pool_feat_size_, 1,
        (Dtype)1.0, bottom_related + i*B_, top_diff + i*pool_feat_size_, (Dtype)0.0,
        diff_feat_sig_grad + i*B_*pool_feat_size_);
  }

  sigmoid_->Backward_cpu(B_*B_*pool_feat_size_, diff_feat_sig, diff_feat_sig_grad,
      diff_feat_grad);
  
  if (this->param_propagate_down_[0]) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, pool_feat_size_, B_*B_, (Dtype)1.0,
        diff_data, diff_feat_grad, (Dtype)0., weight_diff);
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, pool_feat_size_, B_*B_, (Dtype)1.0,
        diff_data + B_*B_, diff_feat_grad, (Dtype)0., weight_diff + pool_feat_size_);
  }


  if (this->param_propagate_down_[1]) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, pool_feat_size_, B_*B_, (Dtype)1.0,
        const_multiplier_.cpu_data(), diff_feat_grad, (Dtype)0., bias_diff);
  }

  if (propagate_down[0]) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_*B_, 1, pool_feat_size_, (Dtype)1.0,
        diff_feat_grad, weight, (Dtype)0., diff_grad);
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_*B_, 1, pool_feat_size_, (Dtype)1.0,
        diff_feat_grad, weight + pool_feat_size_, (Dtype)0., diff_grad + B_*B_);

    caffe_set<Dtype>(B_*feat_dim_, (Dtype)0.0, bottom_diff);

    for (int i = 0; i < B_; ++i) {
      for (int j = 0; j < 2; ++j) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, 1, B_, (Dtype)(-1.0),
            diff_grad + j*B_*B_, eye_.cpu_data() + i*B_, (Dtype)0., temp_B_data);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, B_, B_, (Dtype)(1.0),
            eye_.cpu_data() + i*B_, diff_grad + j*B_*B_, (Dtype)1., temp_B_data);
        bottom_diff[i*feat_dim_ + j] = caffe_cpu_dot<Dtype>(B_, temp_B_data,
            const_multiplier_.cpu_data());
      }
    }
  }
  
  /*int index_x = 0, index_y = 0;

  int half_pool_size = pool_size_/2;
  Dtype half_bin_size = (Dtype)(pool_bin_size_/2.0);

  caffe_set<Dtype>(B_*2, (Dtype)0.0, bottom_diff);


  for (int i = 0; i < B_; ++i) {
    for (int j = 0; j < B_; ++j) {
      if ((*bottom)[1]->data_at(0, i, j, 0) > (Dtype)0.0) {
        index_x = floor((double)(diff_.data_at(0, i, j, 0) + half_bin_size)/ (double)(2* half_bin_size));
        index_x = index_x + half_pool_size;
        index_y = floor((double)(diff_.data_at(1, i, j, 0) + half_bin_size)/ (double)(2* half_bin_size));
        index_y = index_y + half_pool_size;
        if ((index_x) >= 0 && (index_x) < pool_size_
            &&(index_y) >= 0 && (index_y) < pool_size_) {
          Dtype grad_value = top_diff[i*pool_size_*pool_size_ + index_x*pool_size_ + index_y];
          bottom_diff[i*2 + 0] += grad_value;
          bottom_diff[i*2 + 1] += grad_value;
          bottom_diff[j*2 + 0] -= grad_value;
          bottom_diff[j*2 + 1] -= grad_value;
        }
      }
    }
  }*/

}


#ifdef CPU_ONLY
STUB_GPU(SocialPoolingLayer);
#endif

INSTANTIATE_CLASS(SocialPoolingLayer);


}  // namespace caffe
