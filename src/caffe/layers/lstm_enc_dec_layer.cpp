#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

/*
 bottom[0] - (temporal_size * num_batch * input_dimension * 1)
 bottom[1] - (temporal_size * num_batch * 1 * 1)
 bottom[2] - (temporal_size * num_batch * input_dimension * 1)
 bottom[3] - (temporal_size * num_batch * 1 * 1)
 top[0] - (temporal_size * num_batch * output_dimension * 1)
 top[1] - (temporal_size * num_batch * output_dimension * 1)
*/

template <typename Dtype>
void LstmEncDecLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  encoder_lstm_->LayerSetUp(bottom[0]);
  encoder_lstm_->SetAsEncoder(true);

  decoder_lstm_->LayerSetUp(bottom[2]);
  decoder_lstm_->SetAsEncoder(false);

  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  const int num_output = this->layer_param_.inner_product_param().num_output();

  H_ = num_output; // number of hidden units
  T_ = bottom[0]->num();
  B_ = bottom[0]->channels();
  I_ = bottom[0]->count() / (T_*B_); // input dimension

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(6);
    } else {
      this->blobs_.resize(4);
    }

    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
 
    // input-to-hidden weights
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(1, 4, H_, I_));
    weight_filler->Fill(this->blobs_[0].get());

    // hidden-to-hidden weights
    // Intialize the weight
    this->blobs_[1].reset(new Blob<Dtype>(1, 4, H_, H_));
    weight_filler->Fill(this->blobs_[1].get());

    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      this->blobs_[4].reset(new Blob<Dtype>(1, 1, 4, H_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[4].get());
 
      caffe_set<Dtype>(H_, Dtype(5.), 
          this->blobs_[4]->mutable_cpu_data() + 1*H_);
    }

    // input-to-hidden weights
    // Intialize the weight
    this->blobs_[2].reset(new Blob<Dtype>(1, 4, H_, I_));
    weight_filler->Fill(this->blobs_[2].get());

    // hidden-to-hidden weights
    // Intialize the weight
    this->blobs_[3].reset(new Blob<Dtype>(1, 4, H_, H_));
    weight_filler->Fill(this->blobs_[3].get());

    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      this->blobs_[5].reset(new Blob<Dtype>(1, 1, 4, H_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[5].get());
 
      caffe_set<Dtype>(H_, Dtype(5.), 
          this->blobs_[5]->mutable_cpu_data() + 1*H_);
    }

  }  // parameter initialization

  this->param_propagate_down_.resize(this->blobs_.size(), true);

}

template <typename Dtype>
void LstmEncDecLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Figure out the dimensions
  //T_ = this->layer_param_.lstm_layer_param().temporal_size();;
  CHECK_EQ(bottom[1]->num(), bottom[0]->num());
  CHECK_EQ(bottom[2]->num(), bottom[3]->num());

  CHECK_EQ(bottom[1]->channels(), bottom[0]->channels());
  CHECK_EQ(bottom[2]->channels(), bottom[0]->channels());
  CHECK_EQ(bottom[3]->channels(), bottom[0]->channels());

  encoder_lstm_->Reshape(bottom[0], bottom[1],  (*top)[0]);
  decoder_lstm_->Reshape(bottom[2], bottom[3], (*top)[1]);
}

template <typename Dtype>
void LstmEncDecLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  const Dtype* weight_i_e = this->blobs_[0]->cpu_data();
  const Dtype* weight_h_e = this->blobs_[1]->cpu_data();
  const Dtype* weight_i_d = this->blobs_[2]->cpu_data();
  const Dtype* weight_h_d = this->blobs_[3]->cpu_data();

  if (bias_term_) {
    const Dtype* bias_e = this->blobs_[4]->cpu_data();
    encoder_lstm_->Forward_cpu(bottom[0]->cpu_data(), bottom[1]->cpu_data(),
      (*top)[0]->mutable_cpu_data(), weight_i_e, weight_h_e, bias_e);
  } else {
    Dtype* bias_e = NULL; 
    encoder_lstm_->Forward_cpu(bottom[0]->cpu_data(), bottom[1]->cpu_data(),
      (*top)[0]->mutable_cpu_data(), weight_i_e, weight_h_e, bias_e);
  }

  Dtype* decoder_cell = decoder_lstm_->next_cell_.mutable_cpu_data();
  Dtype* decoder_out  = decoder_lstm_->next_out_.mutable_cpu_data();

  caffe_copy<Dtype>(B_*H_, encoder_lstm_->next_cell_.cpu_data(), decoder_cell);
  caffe_set<Dtype>(B_*H_, (Dtype)0., decoder_out);

  if (bias_term_) {
    const Dtype* bias_d = this->blobs_[5]->cpu_data();
    decoder_lstm_->Forward_cpu(bottom[2]->cpu_data(), bottom[3]->cpu_data(),
      (*top)[1]->mutable_cpu_data(), weight_i_d, weight_h_d, bias_d);
  } else {
    const Dtype* bias_d = NULL;
    decoder_lstm_->Forward_cpu(bottom[2]->cpu_data(), bottom[3]->cpu_data(),
      (*top)[1]->mutable_cpu_data(), weight_i_d, weight_h_d, bias_d);
  }

}

template <typename Dtype>
void LstmEncDecLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {

  const Dtype* weight_i_e = this->blobs_[0]->cpu_data();
  const Dtype* weight_h_e = this->blobs_[1]->cpu_data();
  const Dtype* weight_i_d = this->blobs_[2]->cpu_data();
  const Dtype* weight_h_d = this->blobs_[3]->cpu_data();

  Dtype* weight_i_e_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* weight_h_e_diff = this->blobs_[1]->mutable_cpu_diff();
  Dtype* weight_i_d_diff = this->blobs_[2]->mutable_cpu_diff();
  Dtype* weight_h_d_diff = this->blobs_[3]->mutable_cpu_diff();

  if (bias_term_) {
    Dtype* bias_d_diff = this->blobs_[5]->mutable_cpu_diff();
    decoder_lstm_->Backward_cpu(top[1]->cpu_data(),
      top[1]->mutable_cpu_diff(), propagate_down[2],
      (*bottom)[2]->cpu_data(), (*bottom)[3]->cpu_data(),
      (*bottom)[2]->mutable_cpu_diff(),
      weight_i_d, weight_h_d,
      weight_i_d_diff, weight_h_d_diff, bias_d_diff);
  } else {
    Dtype* bias_d_diff = NULL;
    decoder_lstm_->Backward_cpu(top[1]->cpu_data(),
      top[1]->mutable_cpu_diff(), propagate_down[2],
      (*bottom)[2]->cpu_data(), (*bottom)[3]->cpu_data(),
      (*bottom)[2]->mutable_cpu_diff(),
      weight_i_d, weight_h_d,
      weight_i_d_diff, weight_h_d_diff, bias_d_diff);
  }

  caffe_copy<Dtype>(B_*H_, decoder_lstm_->next_cell_diff_.cpu_data(),
      encoder_lstm_->next_cell_diff_.mutable_cpu_data());

  if (bias_term_) {
    Dtype* bias_e_diff = this->blobs_[4]->mutable_cpu_diff();
    encoder_lstm_->Backward_cpu(top[0]->cpu_data(),
      top[1]->mutable_cpu_diff(), propagate_down[0],
      (*bottom)[0]->cpu_data(), (*bottom)[1]->cpu_data(),
      (*bottom)[0]->mutable_cpu_diff(),
      weight_i_e, weight_h_e,
      weight_i_e_diff, weight_h_e_diff, bias_e_diff);
  } else {
    Dtype* bias_e_diff = NULL;
    encoder_lstm_->Backward_cpu(top[0]->cpu_data(),
      top[1]->mutable_cpu_diff(), propagate_down[0],
      (*bottom)[0]->cpu_data(), (*bottom)[1]->cpu_data(),
      (*bottom)[0]->mutable_cpu_diff(),
      weight_i_e, weight_h_e,
      weight_i_e_diff, weight_h_e_diff, bias_e_diff);
  }

}

#ifdef CPU_ONLY
STUB_GPU(LstmEncDecLayer);
#endif

INSTANTIATE_CLASS(LstmEncDecLayer);

}  // namespace caffe

