#include <algorithm>
#include <vector>

#include "caffe/vision_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// Pool in a region around each trajectory position 
template <typename Dtype>
void SocialPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* mean_added_data = NULL;
  if (bottom.size() > 2) {
    mean_added_data = mean_added_data_.mutable_gpu_data();
    //caffe_gpu_add<Dtype>(B_*feat_dim_, bottom_data, bottom[2]->gpu_data(), mean_added_data);
    caffe_copy<Dtype>(B_*feat_dim_, bottom_data, mean_added_data);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, feat_dim_, 2, (Dtype)1.,
        bottom[2]->gpu_data(), f22_mat_.gpu_data(), (Dtype)1., mean_added_data);
  }
  
  const Dtype* bottom_related = bottom[1]->gpu_data();

  Dtype* diff_data = diff_.mutable_gpu_data();
  const Dtype* x_selector = x_sel_.gpu_data();
  const Dtype* y_selector = y_sel_.gpu_data();

  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  Dtype* temp_B_data = temp_B_.mutable_gpu_data();

  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* bias = this->blobs_[1]->gpu_data();

  Dtype* diff_feat = diff_feat_.mutable_gpu_data();
  Dtype* diff_feat_sig = diff_feat_sig_.mutable_gpu_data();

  // First get the x-difference in diff_data
  //
  if (bottom.size() > 2) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, B_, feat_dim_, (Dtype)1., mean_added_data,
        x_selector, (Dtype)0.0, temp_B_data); 
  } else {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, B_, feat_dim_, (Dtype)1., bottom_data,
        x_selector, (Dtype)0.0, temp_B_data);
  }

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, B_, 1, (Dtype)1.0,
      temp_B_data, const_multiplier_.gpu_data(), (Dtype)0.0, diff_data);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, B_, 1, (Dtype)(-1.0),
      const_multiplier_.gpu_data(), temp_B_data, (Dtype)1.0, diff_data);

  // First get the y-difference in diff_data + (B_*B_)
  if (bottom.size() > 2) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, B_, feat_dim_, (Dtype)1., mean_added_data,
        y_selector, (Dtype)0.0, temp_B_data);
  } else {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, B_, feat_dim_, (Dtype)1., bottom_data,
        y_selector, (Dtype)0.0, temp_B_data);
  }

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, B_, 1, (Dtype)1.0,
      temp_B_data, const_multiplier_.gpu_data(), (Dtype)0.0, diff_data + B_*B_);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, B_, 1, (Dtype)(-1.0),
      const_multiplier_.gpu_data(), temp_B_data, (Dtype)1.0, diff_data + B_*B_);

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_*B_, pool_feat_size_, 1, (Dtype)1.0,
    diff_data, weight, (Dtype)0.0, diff_feat);

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_*B_, pool_feat_size_, 1, (Dtype)1.0,
    diff_data + B_*B_, weight + pool_feat_size_, (Dtype)1.0, diff_feat);

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_*B_, pool_feat_size_, 1, (Dtype)1.0,
    const_multiplier_.gpu_data(), bias, (Dtype)1.0, diff_feat);

  sigmoid_->Forward_gpu(B_*B_*pool_feat_size_, diff_feat, diff_feat_sig);
  // Find the bucket into which each of the values fall (Only max pooling for now)
  //caffe_gpu_set<Dtype>(B_*pool_feat_size_, (Dtype)0.0, top_data);
  
  for (int i = 0; i < B_; ++i) {
    caffe_gpu_gemv<Dtype>(CblasTrans, pool_feat_size_, B_, (Dtype)1.0,
        diff_feat_sig + (i*B_*pool_feat_size_), bottom_related + i*B_, 
        (Dtype)0.0, top_data + i*pool_feat_size_);
  }

}

template <typename Dtype>
void SocialPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  //Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* bottom_mutable_diff = (*bottom)[0]->mutable_gpu_diff();

  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  //caffe_gpu_set((*bottom)[0]->count(), Dtype(0), bottom_diff);

  //const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  const Dtype* bottom_related = (*bottom)[1]->gpu_data();

  const Dtype* diff_data = diff_.gpu_data();
  //const Dtype* diff_feat = diff_feat_.gpu_data();
  const Dtype* diff_feat_sig = diff_feat_sig_.gpu_data();

  Dtype* diff_grad = diff_.mutable_gpu_diff();
  Dtype* diff_feat_sig_grad = diff_feat_sig_.mutable_gpu_diff();
  Dtype* diff_feat_grad = diff_feat_.mutable_gpu_diff();

  Dtype* temp_B_data = temp_B_.mutable_gpu_data();

  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();

  const Dtype* weight = this->blobs_[0]->gpu_data();
  //const Dtype* bias = this->blobs_[1]->gpu_data();

  for (int i = 0; i < B_; ++i) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, pool_feat_size_, 1,
        (Dtype)1.0, bottom_related + i*B_, top_diff + i*pool_feat_size_, (Dtype)0.0,
        diff_feat_sig_grad + i*B_*pool_feat_size_);
  }

  sigmoid_->Backward_gpu(B_*B_*pool_feat_size_, diff_feat_sig, diff_feat_sig_grad,
      diff_feat_grad);
  
  if (this->param_propagate_down_[0]) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, pool_feat_size_, B_*B_, (Dtype)1.0,
        diff_data, diff_feat_grad, (Dtype)0., weight_diff);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, pool_feat_size_, B_*B_, (Dtype)1.0,
        diff_data + B_*B_, diff_feat_grad, (Dtype)0., weight_diff + pool_feat_size_);
  }


  if (this->param_propagate_down_[1]) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, pool_feat_size_, B_*B_, (Dtype)1.0,
        const_multiplier_.gpu_data(), diff_feat_grad, (Dtype)0., bias_diff);
  }

  //return;

  if (propagate_down[0]) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_*B_, 1, pool_feat_size_, (Dtype)1.0,
        diff_feat_grad, weight, (Dtype)0., diff_grad);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_*B_, 1, pool_feat_size_, (Dtype)1.0,
        diff_feat_grad, weight + pool_feat_size_, (Dtype)0., diff_grad + B_*B_);

    //caffe_set<Dtype>(B_*feat_dim_, (Dtype)0.0, bottom_diff);

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, 1, B_, (Dtype)(1.0),
            diff_grad, const_multiplier_.gpu_data(), (Dtype)0., temp_B_data);
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, B_, 1, B_, (Dtype)(-1.0),
            diff_grad, const_multiplier_.gpu_data(), (Dtype)1., temp_B_data);
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, B_, feat_dim_, 1, (Dtype)(1.0),
            temp_B_data, f22_mat_.gpu_data(), (Dtype)0., bottom_mutable_diff);

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, 1, B_, (Dtype)(1.0),
            diff_grad + B_*B_, const_multiplier_.gpu_data(), (Dtype)0., temp_B_data);
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, B_, 1, B_, (Dtype)(-1.0),
            diff_grad + B_*B_, const_multiplier_.gpu_data(), (Dtype)1., temp_B_data);
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, B_, feat_dim_, 1, (Dtype)(1.0),
            temp_B_data, f22_mat_.gpu_data() + feat_dim_, (Dtype)1., bottom_mutable_diff);


    /*for (int i = 0; i < B_; ++i) {
      for (int j = 0; j < 2; ++j) {
        Dtype dot_value;
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, 1, B_, (Dtype)(-1.0),
            diff_grad + j*B_*B_, eye_.gpu_data() + i*B_, (Dtype)0., temp_B_data);
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, B_, B_, (Dtype)(1.0),
            eye_.gpu_data() + i*B_, diff_grad + j*B_*B_, (Dtype)1., temp_B_data);
        caffe_gpu_dot<Dtype>(B_, temp_B_data,
            const_multiplier_.gpu_data(), &dot_value);
        bottom_diff[i*feat_dim_ + j] = dot_value;
      }
    }*/
  }
  
}


INSTANTIATE_CLASS(SocialPoolingLayer);


}  // namespace caffe
