#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SumLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  num_output_ = this->layer_param_.sum_param().num_output();

  (*top)[0]->Reshape(bottom[0]->num(), num_output_, 1, 1);

  temp_.Reshape(bottom[0]->num(), 1, 1, 1);

  sum_multiplier_.Reshape(1, bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);

  sum_multiplier_2_.Reshape(1, num_output_, 1, 1);
  Dtype* multiplier_data_2 = sum_multiplier_2_.mutable_cpu_data();
  caffe_set(sum_multiplier_2_.count(), Dtype(1), multiplier_data_2);

}

template <typename Dtype>
void SumLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;

  if (num_output_ == 1) {
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1, bottom_data,
        sum_multiplier_.cpu_data(), 0., top_data);  // summer
  } else {
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1, bottom_data,
        sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());  // summer
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, num_output_, 1, 1,
        temp_.cpu_data(), sum_multiplier_2_.cpu_data(), 0., top_data);  // summer

    /*for (int i = 0; i < num; ++i) {
      caffe_set(num_output_, temp_.cpu_data()[i], top_data + (*top)[0]->offset(i));
    }*/
  }

}

template <typename Dtype>
void SumLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  int num = (*bottom)[0]->num();
  int dim = (*bottom)[0]->count() / num;

  if (num_output_ == 1) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1,
        top_diff, sum_multiplier_.cpu_data(), 0., bottom_diff);
    /*for (int i = 0; i < num; ++i) {
      caffe_set(dim, top_diff[i], bottom_diff + (*bottom)[0]->offset(i));
    }*/
  } else {
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, num_output_, 1, top_diff,
        sum_multiplier_2_.cpu_data(), 0., temp_.mutable_cpu_data());  // summer
    
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1,
        temp_.cpu_data(), sum_multiplier_.cpu_data(), 0., bottom_diff);  // summer

    /*for (int i = 0; i < num; ++i) {
      caffe_set(dim, temp_.cpu_data()[i], bottom_diff + (*bottom)[0]->offset(i));
    }*/
  }
}


#ifdef CPU_ONLY
STUB_GPU(SumLayer);
#endif

INSTANTIATE_CLASS(SumLayer);


}  // namespace caffe
