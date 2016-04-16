#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void FlattenBatchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  int channels_out =  bottom[0]->height()
      * bottom[0]->width();

  int batch_out = bottom[0]->channels() * bottom[0]->num();
  int feat_size = 1;

  if (this->layer_param_.flatten_batch_param().batch_size() > 0) {
    batch_out = this->layer_param_.flatten_batch_param().batch_size();
    channels_out = bottom[0]->num()/this->layer_param_.flatten_batch_param().batch_size();
    feat_size = bottom[0]->width() * bottom[0]->height() * bottom[0]->channels();
  }

  (*top)[0]->Reshape(batch_out, channels_out, feat_size, 1);
  count_ = batch_out * channels_out * feat_size;
  CHECK_EQ(count_, bottom[0]->count());
  CHECK_EQ(count_, (*top)[0]->count());
}

template <typename Dtype>
void FlattenBatchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  (*top)[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void FlattenBatchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  (*bottom)[0]->ShareDiff(*top[0]);
}

#ifdef CPU_ONLY
STUB_GPU(FlattenBatchLayer);
#endif

INSTANTIATE_CLASS(FlattenBatchLayer);

}  // namespace caffe
