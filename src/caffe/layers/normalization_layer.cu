#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NormalizationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  int num;
  num = bottom[0]->num();

  int dim = bottom[0]->count() / num;

  Dtype eps = 1e-10;

  // put the squares of bottom into temp_
  caffe_gpu_powx(bottom[0]->count(), bottom_data, Dtype(2),
      temp_.mutable_gpu_data());

  // Sum of the squares
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1, temp_.gpu_data(),
      sum_multiplier_.gpu_data(), 0., sum_square_.mutable_gpu_data());

  // Sqrt
  caffe_gpu_powx(sum_square_.count(), sum_square_.gpu_data(), Dtype(0.5),
        sum_square_.mutable_gpu_data());

  // Add eps to avoid zero-division
  caffe_gpu_add_scalar(sum_square_.count(), eps, sum_square_.mutable_gpu_data());

  // replicate the value along the feature dimension
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1,
      sum_square_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
      temp_.mutable_gpu_data());

  // Finally divide all values by the normalization value
  caffe_gpu_div(temp_.count(), bottom_data, temp_.gpu_data(), top_data);

}

template <typename Dtype>
void NormalizationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();

  int num = (*bottom)[0]->num();
  int dim = (*bottom)[0]->count() / num;

  Dtype eps = 1e-10;

  // x_b^T * d_t^T
  caffe_gpu_mul(temp_.count(), bottom_data, top_diff, temp_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1, temp_.gpu_data(),
      sum_multiplier_.gpu_data(), 0., sum_square_.mutable_gpu_data());

  // replicate the value along the feature dimension
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1,
      sum_square_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
      bottom_diff);
  caffe_gpu_mul(temp_.count(), bottom_data, bottom_diff, bottom_diff);

  // put the squares of bottom into temp_
  caffe_gpu_powx(temp_.count(), bottom_data, Dtype(2),
      temp_.mutable_gpu_data());

  // Sum of the squares
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1, temp_.gpu_data(),
      sum_multiplier_.gpu_data(), 0., sum_square_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1,
      sum_square_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
      temp_.mutable_gpu_data());
  caffe_gpu_mul(temp_.count(), temp_.gpu_data(), top_diff, temp2_.mutable_gpu_data());

  // Get numerator
  caffe_gpu_sub(temp_.count(), temp2_.gpu_data(), bottom_diff, bottom_diff);

  // Divide to normalize
  caffe_gpu_powx(temp_.count(), temp_.gpu_data(), Dtype(1.5),
      temp_.mutable_gpu_data());

  // Add eps to avoid zero-division
  caffe_gpu_add_scalar(temp_.count(), eps, temp_.mutable_gpu_data());

  // Finally divide all values by the normalization value
  caffe_gpu_div(temp_.count(), bottom_diff, temp_.gpu_data(), bottom_diff);

}


INSTANTIATE_CLASS(NormalizationLayer);


}  // namespace caffe
