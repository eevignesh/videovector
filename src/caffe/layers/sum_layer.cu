#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SumLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  //Forward_cpu(bottom, top); return;

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;

  if (num_output_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1, bottom_data,
        sum_multiplier_.gpu_data(), 0., top_data);  // summer
  } else {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1, bottom_data,
        sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());  // summer
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, num_output_, 1, 1,
        temp_.gpu_data(), sum_multiplier_2_.gpu_data(), 0., top_data);  // summer
  }

}

template <typename Dtype>
void SumLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {

  //Backward_cpu(top, propagate_down, bottom); return;

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();

  int num = (*bottom)[0]->num();
  int dim = (*bottom)[0]->count() / num;

  if (num_output_ == 1) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1,
        top_diff, sum_multiplier_.gpu_data(), 0., bottom_diff);
  } else {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num, num_output_, 1, top_diff,
        sum_multiplier_2_.gpu_data(), 0., temp_.mutable_gpu_data());  // summer
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1,
        temp_.gpu_data(), sum_multiplier_.gpu_data(), 0., bottom_diff);  // summer
  }
}


INSTANTIATE_CLASS(SumLayer);


}  // namespace caffe
