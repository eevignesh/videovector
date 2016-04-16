#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void LstmEncDecLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  const Dtype* weight_i_e = this->blobs_[0]->gpu_data();
  const Dtype* weight_h_e = this->blobs_[1]->gpu_data();
  const Dtype* weight_i_d = this->blobs_[2]->gpu_data();
  const Dtype* weight_h_d = this->blobs_[3]->gpu_data();

  if (bias_term_) {
    const Dtype* bias_e = this->blobs_[4]->gpu_data();
    encoder_lstm_->Forward_gpu(bottom[0]->gpu_data(), bottom[1]->gpu_data(),
      (*top)[0]->mutable_gpu_data(), weight_i_e, weight_h_e, bias_e);
  } else {
    Dtype* bias_e = NULL; 
    encoder_lstm_->Forward_gpu(bottom[0]->gpu_data(), bottom[1]->gpu_data(),
      (*top)[0]->mutable_gpu_data(), weight_i_e, weight_h_e, bias_e);
  }

  Dtype* decoder_cell = decoder_lstm_->next_cell_.mutable_gpu_data();
  Dtype* decoder_out  = decoder_lstm_->next_out_.mutable_gpu_data();

  caffe_copy<Dtype>(B_*H_, encoder_lstm_->next_cell_.gpu_data(), decoder_cell);
  caffe_gpu_set<Dtype>(B_*H_, (Dtype)0., decoder_out);

  if (bias_term_) {
    const Dtype* bias_d = this->blobs_[5]->gpu_data();
    decoder_lstm_->Forward_gpu(bottom[2]->gpu_data(), bottom[3]->gpu_data(),
      (*top)[1]->mutable_gpu_data(), weight_i_d, weight_h_d, bias_d);
  } else {
    const Dtype* bias_d = NULL;
    decoder_lstm_->Forward_gpu(bottom[2]->gpu_data(), bottom[3]->gpu_data(),
      (*top)[1]->mutable_gpu_data(), weight_i_d, weight_h_d, bias_d);
  }

}

template <typename Dtype>
void LstmEncDecLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {

  const Dtype* weight_i_e = this->blobs_[0]->gpu_data();
  const Dtype* weight_h_e = this->blobs_[1]->gpu_data();
  const Dtype* weight_i_d = this->blobs_[2]->gpu_data();
  const Dtype* weight_h_d = this->blobs_[3]->gpu_data();

  Dtype* weight_i_e_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* weight_h_e_diff = this->blobs_[1]->mutable_gpu_diff();
  Dtype* weight_i_d_diff = this->blobs_[2]->mutable_gpu_diff();
  Dtype* weight_h_d_diff = this->blobs_[3]->mutable_gpu_diff();


  if (bias_term_) {
    Dtype* bias_d_diff = this->blobs_[5]->mutable_gpu_diff();
    decoder_lstm_->Backward_gpu(top[1]->gpu_data(),
      top[1]->mutable_gpu_diff(), propagate_down[2],
      (*bottom)[2]->gpu_data(), (*bottom)[3]->gpu_data(),
      (*bottom)[2]->mutable_gpu_diff(),
      weight_i_d, weight_h_d,
      weight_i_d_diff, weight_h_d_diff, bias_d_diff);
  } else {
    Dtype* bias_d_diff = NULL;
    decoder_lstm_->Backward_gpu(top[1]->gpu_data(),
      top[1]->mutable_gpu_diff(), propagate_down[2],
      (*bottom)[2]->gpu_data(), (*bottom)[3]->gpu_data(),
      (*bottom)[2]->mutable_gpu_diff(),
      weight_i_d, weight_h_d,
      weight_i_d_diff, weight_h_d_diff, bias_d_diff);
  }

  caffe_copy<Dtype>(B_*H_, decoder_lstm_->next_cell_diff_.gpu_data(),
      encoder_lstm_->next_cell_diff_.mutable_gpu_data());

  if (bias_term_) {
    Dtype* bias_e_diff = this->blobs_[4]->mutable_gpu_diff();
    encoder_lstm_->Backward_gpu(top[0]->gpu_data(),
      top[1]->mutable_gpu_diff(), propagate_down[0],
      (*bottom)[0]->gpu_data(), (*bottom)[1]->gpu_data(),
      (*bottom)[0]->mutable_gpu_diff(),
      weight_i_e, weight_h_e,
      weight_i_e_diff, weight_h_e_diff, bias_e_diff);
  } else {
    Dtype* bias_e_diff = NULL;
    encoder_lstm_->Backward_gpu(top[0]->gpu_data(),
      top[1]->mutable_gpu_diff(), propagate_down[0],
      (*bottom)[0]->gpu_data(), (*bottom)[1]->gpu_data(),
      (*bottom)[0]->mutable_gpu_diff(),
      weight_i_e, weight_h_e,
      weight_i_e_diff, weight_h_e_diff, bias_e_diff);
  }

}




INSTANTIATE_CLASS(LstmEncDecLayer);

}  // namespace caffe

