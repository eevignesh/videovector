#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());

  diff_abs_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());

  sum_mult_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  caffe_set(bottom[0]->count(), (Dtype)1., sum_mult_.mutable_cpu_data());

}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());

  if (this->layer_param_.euclidean_loss_param().l1()) {
  
    caffe_abs(count, diff_.cpu_data(), diff_abs_.mutable_cpu_data());
    Dtype loss = caffe_cpu_dot(count, diff_abs_.cpu_data(), sum_mult_.cpu_data());
    loss = loss / bottom[0]->num();
    (*top)[0]->mutable_cpu_data()[0] = loss;

  } else {
    Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
    Dtype loss = dot / bottom[0]->num() / Dtype(2);
    (*top)[0]->mutable_cpu_data()[0] = loss;
  }
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {

  if (this->layer_param_.euclidean_loss_param().l1()) {
    for (int c = 0; c < (*bottom)[0]->count(); ++c) {
      const Dtype* data_0 = (*bottom)[0]->cpu_data() + c;
      const Dtype* data_1 = (*bottom)[1]->cpu_data() + c;
      if ( *(data_0) > (*(data_1) + 1e-6) ) {
        if (propagate_down[0]) {
          *(top[0]->mutable_cpu_diff() + c) = (Dtype)(1.0/(*bottom)[0]->num());
        }
        if (propagate_down[1]) {
          *(top[1]->mutable_cpu_diff() + c) = (Dtype)(-1.0/(*bottom)[0]->num());
        }

      } else if ( *(data_0) < (*(data_1) + 1e-6) ) {

        if (propagate_down[0]) {
          *(top[0]->mutable_cpu_diff() + c) = (Dtype)(-1.0/(*bottom)[0]->num());
        }
        if (propagate_down[1]) {
          *(top[1]->mutable_cpu_diff() + c) = (Dtype)(1.0/(*bottom)[0]->num());
        }
      } else {

        if (propagate_down[0]) {
          *(top[0]->mutable_cpu_diff() + c) = (Dtype)(0.0);
        }
        if (propagate_down[1]) {
          *(top[1]->mutable_cpu_diff() + c) = (Dtype)(0.0);
        }
      } 
    }
  } else {
    for (int i = 0; i < 2; ++i) {
      if (propagate_down[i]) {
          const Dtype sign = (i == 0) ? 1 : -1;
          const Dtype alpha = sign * top[0]->cpu_diff()[0] / (*bottom)[i]->num();
          caffe_cpu_axpby(
              (*bottom)[i]->count(),              // count
              alpha,                              // alpha
              diff_.cpu_data(),                   // a
              Dtype(0),                           // beta
              (*bottom)[i]->mutable_cpu_diff());  // b
        }
      }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);

}  // namespace caffe
