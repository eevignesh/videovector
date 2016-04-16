#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NormalizationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());

  sum_square_.Reshape(bottom[0]->num(), 1,
      1, 1);
  temp_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  temp2_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  sum_multiplier_.Reshape(1, bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());

  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
}

template <typename Dtype>
void NormalizationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;
  Dtype eps = 1e-10;

  // put the squares of bottom into temp_
  caffe_powx(bottom[0]->count(), bottom_data, Dtype(2),
    temp_.mutable_cpu_data());

  // Sum of the squares
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1, temp_.cpu_data(),
      sum_multiplier_.cpu_data(), 0., sum_square_.mutable_cpu_data());

  // Sqrt
  caffe_powx(sum_square_.count(), sum_square_.cpu_data(), Dtype(0.5),
        sum_square_.mutable_cpu_data());

  // Add eps to avoid zero-division
  caffe_add_scalar(sum_square_.count(), eps, sum_square_.mutable_cpu_data());

  // replicate the value along the feature dimension
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1,
      sum_square_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());

  // Finally divide all values by the normalization value
  caffe_div(temp_.count(), bottom_data, temp_.cpu_data(), top_data);

}

template <typename Dtype>
void NormalizationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  //const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();

  int num = (*bottom)[0]->num();
  int dim = (*bottom)[0]->count() / num;
  Dtype eps = 1e-10;

  // x_b^T * d_t^T
  caffe_mul(temp_.count(), bottom_data, top_diff, temp_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1, temp_.mutable_cpu_data(),
      sum_multiplier_.cpu_data(), 0., sum_square_.mutable_cpu_data());

  // replicate the value along the feature dimension
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1,
      sum_square_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      bottom_diff);
  caffe_mul(temp_.count(), bottom_data, bottom_diff, bottom_diff);

  // put the squares of bottom into temp_
  caffe_powx((*bottom)[0]->count(), bottom_data, Dtype(2),
      temp_.mutable_cpu_data());

  // Sum of the squares
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1, temp_.cpu_data(),
      sum_multiplier_.cpu_data(), 0., sum_square_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1,
      sum_square_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_mul(temp_.count(), temp_.cpu_data(), top_diff, temp2_.mutable_cpu_data());

  // Get numerator
  caffe_sub(temp_.count(), temp2_.cpu_data(), bottom_diff, bottom_diff);

  // Divide to normalize
  caffe_powx((*bottom)[0]->count(), temp_.cpu_data(), Dtype(1.5),
      temp_.mutable_cpu_data());

  // Add eps to avoid zero-division
  caffe_add_scalar(temp_.count(), eps, temp_.mutable_cpu_data());

  // Finally divide all values by the normalization value
  caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);

}


#ifdef CPU_ONLY
STUB_GPU(NormalizationLayer);
#endif

INSTANTIATE_CLASS(NormalizationLayer);


}  // namespace caffe
