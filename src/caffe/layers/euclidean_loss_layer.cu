#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();

  /*for (int i = 0; i < 40; ++i) {
    LOG(INFO) << "t: " << i
      << " : " << bottom[0]->data_at(i, 0, 0, 0)
      << " : " << bottom[0]->data_at(i, 1, 0, 0);
  }*/

  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  if (this->layer_param_.euclidean_loss_param().l1()) {
  
    caffe_gpu_abs(count, diff_.gpu_data(), diff_abs_.mutable_gpu_data());
    Dtype loss;
    caffe_gpu_dot(count, diff_abs_.gpu_data(), sum_mult_.gpu_data(), &loss);
    loss = loss / bottom[0]->num();
    (*top)[0]->mutable_cpu_data()[0] = loss;
  } else {
    Dtype dot;
    caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
    Dtype loss = dot / bottom[0]->num() / Dtype(2);
    (*top)[0]->mutable_cpu_data()[0] = loss;
  }
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {

  if (this->layer_param_.euclidean_loss_param().l1()) {
    for (int c = 0; c < (*bottom)[0]->count(); ++c) {
      const Dtype data_0 = (*bottom)[0]->cpu_data()[c];
      const Dtype data_1 = (*bottom)[1]->cpu_data()[c];
      if ( data_0 > (data_1 + 1e-3) ) {
        if (propagate_down[0]) {
          *((*bottom)[0]->mutable_cpu_diff() + c) = (Dtype)(1.0/(*bottom)[0]->num())*top[0]->cpu_diff()[0];
        }
        if (propagate_down[1]) {
          (*bottom)[1]->mutable_cpu_diff()[c] = (Dtype)(-1.0/(*bottom)[0]->num())*top[0]->cpu_diff()[0] ;
        }

      } else if (data_0 < (data_1 + 1e-3) ) {

        if (propagate_down[0]) {
          (*bottom)[0]->mutable_cpu_diff()[c] = (Dtype)(-1.0/(*bottom)[0]->num())*top[0]->cpu_diff()[0] ;
        }
        if (propagate_down[1]) {
          (*bottom)[1]->mutable_cpu_diff()[c] = (Dtype)(1.0/(*bottom)[0]->num())*top[0]->cpu_diff()[0] ;
        }
      } else {

        if (propagate_down[0]) {
          (*bottom)[0]->mutable_cpu_diff()[c] = (Dtype)(0.0);
        }
        if (propagate_down[1]) {
          (*bottom)[1]->mutable_cpu_diff()[c] = (Dtype)(0.0);
        }
      } 
    }
  } else {

    for (int i = 0; i < 2; ++i) {
      if (propagate_down[i]) {
        const Dtype sign = (i == 0) ? 1 : -1;
        const Dtype alpha = sign * top[0]->cpu_diff()[0] / (*bottom)[i]->num();
        caffe_gpu_axpby(
            (*bottom)[i]->count(),              // count
            alpha,                              // alpha
            diff_.gpu_data(),                   // a
            Dtype(0),                           // beta
            (*bottom)[i]->mutable_gpu_diff());  // b
      }
    }
  }
}

INSTANTIATE_CLASS(EuclideanLossLayer);

}  // namespace caffe
