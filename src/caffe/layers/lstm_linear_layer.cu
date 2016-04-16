#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void LstmLinearLayer<Dtype>::Forward_gpu(const Dtype* bottom_data,
    const Dtype* bottom_cont, Dtype* top_data,
    const Dtype* weight_i, const Dtype* weight_h,
    const Dtype* bias){

  //LOG(ERROR) << "Forward";

  Dtype* pre_gate_data_i = pre_gate_i_.mutable_gpu_data();
  Dtype* pre_gate_data_f = pre_gate_f_.mutable_gpu_data();
  Dtype* pre_gate_data_o = pre_gate_o_.mutable_gpu_data();
  Dtype* pre_gate_data_g = pre_gate_g_.mutable_gpu_data();

  Dtype* gate_data_i = gate_i_.mutable_gpu_data();
  Dtype* gate_data_f = gate_f_.mutable_gpu_data();
  Dtype* gate_data_o = gate_o_.mutable_gpu_data();
  Dtype* gate_data_g = gate_g_.mutable_gpu_data();

  Dtype* cell_data = cell_.mutable_gpu_data();
  //Dtype* tanh_cell_data = tanh_cell_.mutable_gpu_data();

  // Initialize previous state
  caffe_copy(B_*H_, next_cell_.gpu_data(), prev_cell_.mutable_gpu_data());
  caffe_copy(B_*H_, next_out_.gpu_data(), prev_out_.mutable_gpu_data());

  // Compute input to hidden forward propagation
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, B_*T_, H_, I_, (Dtype)1.,
      bottom_data, weight_i, (Dtype)0., pre_gate_data_i);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, B_*T_, H_, I_, (Dtype)1.,
      bottom_data, weight_i + H_*I_, (Dtype)0., pre_gate_data_f);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, B_*T_, H_, I_, (Dtype)1.,
      bottom_data, weight_i + 2*H_*I_, (Dtype)0., pre_gate_data_o);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, B_*T_, H_, I_, (Dtype)1.,
      bottom_data, weight_i + 3*H_*I_, (Dtype)0., pre_gate_data_g);

  // Add bias 
  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_*T_, H_, 1, (Dtype)1.,
        bias_multiplier_.gpu_data(), bias, (Dtype)1., pre_gate_data_i);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_*T_, H_, 1, (Dtype)1.,
        bias_multiplier_.gpu_data(), bias + H_, (Dtype)1., pre_gate_data_f);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_*T_, H_, 1, (Dtype)1.,
        bias_multiplier_.gpu_data(), bias + 2*H_, (Dtype)1., pre_gate_data_o);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_*T_, H_, 1, (Dtype)1.,
        bias_multiplier_.gpu_data(), bias + 3*H_, (Dtype)1., pre_gate_data_g);

  }

  // Compute recurrent forward propagation
  for (int t = 0; t < T_; ++t) {
    Dtype* h_t = top_data + t*B_*H_;
    Dtype* c_t = cell_data + t*B_*H_;
    //Dtype* tanh_c_t = tanh_cell_data + t*B_*H_;

    Dtype* i_t = gate_data_i + t*B_*H_;
    Dtype* f_t = gate_data_f + t*B_*H_; 
    Dtype* o_t = gate_data_o + t*B_*H_; 
    Dtype* g_t = gate_data_g + t*B_*H_;
 
    Dtype* pre_i_t = pre_gate_data_i + t*B_*H_;
    Dtype* pre_f_t = pre_gate_data_f + t*B_*H_;
    Dtype* pre_o_t = pre_gate_data_o + t*B_*H_;
    Dtype* pre_g_t = pre_gate_data_g + t*B_*H_;

    Dtype* ig = ig_.mutable_gpu_data();
    const Dtype* seq_cont = bottom_cont + t*B_;
    // Add hidden-to-hidden propagation
    const Dtype* h_t_1 = t > 0 ? (h_t - B_*H_) : prev_out_.gpu_data();

    // For identifying sequence starter
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, H_, 1, (Dtype)1.,
        seq_cont, cont_multiplier_.gpu_data(), (Dtype)0.,
        cont_values_.mutable_gpu_data());
    caffe_gpu_mul<Dtype>(B_*H_, h_t_1,
        cont_values_.mutable_gpu_data(), temp_bh_.mutable_gpu_data());

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, B_, H_, H_, (Dtype)1.,
        temp_bh_.mutable_gpu_data(), weight_h, (Dtype)1., pre_i_t);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, B_, H_, H_, (Dtype)1.,
        temp_bh_.mutable_gpu_data(), weight_h + H_*H_, (Dtype)1., pre_f_t);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, B_, H_, H_, (Dtype)1.,
        temp_bh_.mutable_gpu_data(), weight_h + 2*H_*H_, (Dtype)1., pre_o_t);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, B_, H_, H_, (Dtype)1.,
        temp_bh_.mutable_gpu_data(), weight_h + 3*H_*H_, (Dtype)1., pre_g_t);
    //caffe_gpu_gemv<Dtype>(CblasNoTrans, 4*H_, H_, (Dtype)1.,
    //    weight_h, h_t_1, (Dtype)1., pre_i_t);

    // Apply nonlinearity
    // Sigmoid - input/forget/output gate
    // TanH - modulation gate
    sigmoid_->Forward_gpu(H_*B_, pre_i_t, i_t);
    sigmoid_->Forward_gpu(H_*B_, pre_f_t, f_t);
    sigmoid_->Forward_gpu(H_*B_, pre_o_t, o_t);

    caffe_copy(H_*B_, pre_g_t, g_t);

    // Compute cell : c(t) = f(t)*c(t-1) + i(t)*g(t)
    const Dtype* c_t_1 = t > 0 ? (c_t - B_*H_) : prev_cell_.gpu_data();
    caffe_gpu_mul<Dtype>(B_*H_, c_t_1,
        cont_values_.mutable_gpu_data(), temp_bh_.mutable_gpu_data());


    caffe_gpu_mul<Dtype>(B_*H_, f_t, temp_bh_.mutable_gpu_data(), c_t);
    caffe_gpu_mul<Dtype>(B_*H_, i_t, g_t, ig);
    caffe_gpu_add<Dtype>(B_*H_, c_t, ig, c_t);

    // Compute output 
    //tanh_->Forward_gpu(B_*H_, c_t, tanh_c_t);
    caffe_gpu_mul<Dtype>(B_*H_, o_t, c_t, h_t);
  }

  // Preserve cell state and output value
  caffe_copy(B_*H_, cell_data + (T_-1)*B_*H_, next_cell_.mutable_gpu_data());
  caffe_copy(B_*H_, top_data + (T_-1)*B_*H_, next_out_.mutable_gpu_data());


}
template <typename Dtype>
void LstmLinearLayer<Dtype>::Backward_gpu(const Dtype* top_data,
    Dtype* top_diff,
    const bool& propagate_down, const Dtype* bottom_data,
    const Dtype* bottom_cont, Dtype* bottom_mutable_diff_data,
    const Dtype* weight_i, const Dtype* weight_h,
    Dtype* weight_i_diff, Dtype* weight_h_diff, Dtype* bias_diff) {

  const Dtype* gate_data_i = gate_i_.gpu_data();
  const Dtype* gate_data_f = gate_f_.gpu_data();
  const Dtype* gate_data_o = gate_o_.gpu_data();
  const Dtype* gate_data_g = gate_g_.gpu_data();

  const Dtype* cell_data = cell_.gpu_data();
  //const Dtype* tanh_cell_data = tanh_cell_.gpu_data();

  Dtype* pre_gate_diff_i = pre_gate_i_.mutable_gpu_diff();
  Dtype* pre_gate_diff_f = pre_gate_f_.mutable_gpu_diff();
  Dtype* pre_gate_diff_o = pre_gate_o_.mutable_gpu_diff();
  Dtype* pre_gate_diff_g = pre_gate_g_.mutable_gpu_diff();

  Dtype* gate_diff_i = gate_i_.mutable_gpu_diff();
  Dtype* gate_diff_f = gate_f_.mutable_gpu_diff();
  Dtype* gate_diff_o = gate_o_.mutable_gpu_diff();
  Dtype* gate_diff_g = gate_g_.mutable_gpu_diff();

  Dtype* cell_diff = cell_.mutable_gpu_diff();

  caffe_gpu_set<Dtype>(4*H_*H_, (Dtype)0., weight_h_diff);

  for (int t = T_-1; t >= 0; --t) {

    Dtype* dh_t = top_diff + t*B_*H_;
    const Dtype* c_t = cell_data + t*B_*H_;
    Dtype* dc_t = cell_diff + t*B_*H_;
    //const Dtype* tanh_c_t = tanh_cell_data + t*B_*H_; 
    const Dtype* i_t = gate_data_i + t*B_*H_;
    const Dtype* f_t = gate_data_f + t*B_*H_; 
    const Dtype* o_t = gate_data_o + t*B_*H_; 
    const Dtype* g_t = gate_data_g + t*B_*H_;

    Dtype* di_t = gate_diff_i + t*B_*H_;
    Dtype* df_t = gate_diff_f + t*B_*H_; 
    Dtype* do_t = gate_diff_o + t*B_*H_; 
    Dtype* dg_t = gate_diff_g + t*B_*H_; 

    Dtype* pre_di_t = pre_gate_diff_i + t*B_*H_;
    Dtype* pre_df_t = pre_gate_diff_f + t*B_*H_;
    Dtype* pre_do_t = pre_gate_diff_o + t*B_*H_;
    Dtype* pre_dg_t = pre_gate_diff_g + t*B_*H_;
    Dtype* fdc = fdc_.mutable_gpu_data();

    const Dtype* seq_cont = bottom_cont + t*B_;


    // Output gate : tanh(c(t)) * h_diff(t)
    caffe_gpu_mul<Dtype>(B_*H_, c_t, dh_t, do_t);

    // Cell state : o(t) * tanh'(c(t)) * h_diff(t) + f(t+1) * c_diff(t+1)
    caffe_gpu_mul<Dtype>(B_*H_, o_t, dh_t, dc_t);
    //tanh_->Backward_gpu(B_*H_, tanh_c_t, dc_t, dc_t);
    if (t < T_-1) {

      // TODO: include start_sequence values here
      caffe_gpu_mul<Dtype>(B_*H_, f_t + B_*H_, dc_t + B_*H_, fdc);
      
      // For identifying sequence starter
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, H_, 1, (Dtype)1.,
          seq_cont + B_, cont_multiplier_.gpu_data(), (Dtype)0.,
          cont_values_.mutable_gpu_data());

      // TODO: Change here ...
      caffe_gpu_mul<Dtype>(B_*H_, fdc, cont_values_.mutable_gpu_data(), temp_bh_.mutable_gpu_data());

      caffe_gpu_add<Dtype>(B_*H_, temp_bh_.mutable_gpu_data(), dc_t, dc_t);
    } else if (encoder_mode_) {
      // If in an encoder mode, the gradients are available to this
      // layer from the next cell
      caffe_gpu_add<Dtype>(B_*H_, next_cell_diff_.gpu_data(), dc_t, dc_t);
    }

    if (t==0 && !encoder_mode_) {
       // TODO: include start_sequence values here
      caffe_gpu_mul<Dtype>(B_*H_, f_t, dc_t, next_cell_diff_.mutable_gpu_data());
    }

    // Forget gate : c(t-1) * c_diff(t)
    const Dtype* c_t_1 = t > 0 ? (c_t - B_*H_) : prev_cell_.gpu_data();
    
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, H_, 1, (Dtype)1.,
        seq_cont, cont_multiplier_.gpu_data(), (Dtype)0.,
        cont_values_.mutable_gpu_data());

    caffe_gpu_mul<Dtype>(B_*H_, c_t_1,
        cont_values_.mutable_gpu_data(), temp_bh_.mutable_gpu_data());

    // TODO: include start_sequence values here
    caffe_gpu_mul<Dtype>(B_*H_, temp_bh_.mutable_gpu_data(), dc_t, df_t);

    // Input gate : g(t) * c_diff(t)
    caffe_gpu_mul<Dtype>(B_*H_, g_t, dc_t, di_t);
    // Input modulation gate : i(t) * c_diff(t)
    caffe_gpu_mul<Dtype>(B_*H_, i_t, dc_t, dg_t);

    // Compute derivate before nonlinearity
    sigmoid_->Backward_gpu(B_*H_, i_t, di_t, pre_di_t);
    sigmoid_->Backward_gpu(B_*H_, f_t, df_t, pre_df_t);
    sigmoid_->Backward_gpu(B_*H_, o_t, do_t, pre_do_t);

    caffe_copy(B_*H_, dg_t, pre_dg_t);

    // Clip deriviates before nonlinearity
    if (clipping_threshold_ > 0.0f) {
      caffe_gpu_bound<Dtype>(B_*H_, pre_di_t, -clipping_threshold_, 
          clipping_threshold_, pre_di_t);
      caffe_gpu_bound<Dtype>(B_*H_, pre_df_t, -clipping_threshold_, 
          clipping_threshold_, pre_df_t);
      caffe_gpu_bound<Dtype>(B_*H_, pre_do_t, -clipping_threshold_, 
          clipping_threshold_, pre_do_t);
      caffe_gpu_bound<Dtype>(B_*H_, pre_dg_t, -clipping_threshold_, 
          clipping_threshold_, pre_dg_t);

    }
    
    if (t > 0) {
      // Backprop output errors to the previous time step
      // TODO: include start sequence values here
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, H_, H_, (Dtype)1.,
        pre_di_t, weight_h, (Dtype)0., ig_.mutable_gpu_data());
      caffe_gpu_mul<Dtype>(B_*H_, ig_.mutable_gpu_data(), cont_values_.mutable_gpu_data(),
          temp_bh_.mutable_gpu_data());
      caffe_gpu_add<Dtype>(B_*H_, dh_t - B_*H_, temp_bh_.mutable_gpu_data(),
          dh_t - B_*H_);
     
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, H_, H_, (Dtype)1.,
        pre_df_t, weight_h + H_*H_, (Dtype)0., ig_.mutable_gpu_data());
      caffe_gpu_mul<Dtype>(B_*H_, ig_.mutable_gpu_data(), cont_values_.mutable_gpu_data(),
          temp_bh_.mutable_gpu_data());
      caffe_gpu_add<Dtype>(B_*H_, dh_t - B_*H_, temp_bh_.mutable_gpu_data(),
          dh_t - B_*H_);

      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, H_, H_, (Dtype)1.,
        pre_do_t, weight_h + 2*H_*H_, (Dtype)0., ig_.mutable_gpu_data());
      caffe_gpu_mul<Dtype>(B_*H_, ig_.mutable_gpu_data(), cont_values_.mutable_gpu_data(),
          temp_bh_.mutable_gpu_data());
      caffe_gpu_add<Dtype>(B_*H_, dh_t - B_*H_, temp_bh_.mutable_gpu_data(),
          dh_t - B_*H_);     

      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, H_, H_, (Dtype)1.,
        pre_dg_t, weight_h + 3*H_*H_, (Dtype)0., ig_.mutable_gpu_data());
      caffe_gpu_mul<Dtype>(B_*H_, ig_.mutable_gpu_data(), cont_values_.mutable_gpu_data(),
          temp_bh_.mutable_gpu_data());
      caffe_gpu_add<Dtype>(B_*H_, dh_t - B_*H_, temp_bh_.mutable_gpu_data(),
          dh_t - B_*H_);
      //caffe_gpu_gemv<Dtype>(CblasTrans, 4*H_, H_, (Dtype)1.,
      //  weight_h, pre_di_t, (Dtype)1., dh_t - H_);

    }

    if (this->param_propagate_down_[1]) {
      caffe_gpu_mul<Dtype>(B_*H_, pre_di_t,
          cont_values_.mutable_gpu_data(), temp_bh_.mutable_gpu_data());
      if (t > 0) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, H_, B_, (Dtype)1.,
          temp_bh_.mutable_gpu_data(), top_data + (t-1)*B_*H_,
          (Dtype)1., weight_h_diff);
      } else {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, H_, B_, (Dtype)1.,
          temp_bh_.mutable_gpu_data(), prev_out_.gpu_data(),
          (Dtype)1., weight_h_diff);
      }

      caffe_gpu_mul<Dtype>(B_*H_, pre_df_t,
          cont_values_.mutable_gpu_data(), temp_bh_.mutable_gpu_data());
      if (t > 0) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, H_, B_, (Dtype)1.,
          temp_bh_.mutable_gpu_data(), top_data + (t-1)*B_*H_,
          (Dtype)1., weight_h_diff + H_*H_);
      } else {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, H_, B_, (Dtype)1.,
          temp_bh_.mutable_gpu_data(), prev_out_.gpu_data(),
          (Dtype)1., weight_h_diff + H_*H_);
      }

      caffe_gpu_mul<Dtype>(B_*H_, pre_do_t,
          cont_values_.mutable_gpu_data(), temp_bh_.mutable_gpu_data());
      if (t > 0) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, H_, B_, (Dtype)1.,
          temp_bh_.mutable_gpu_data(), top_data + (t-1)*B_*H_,
          (Dtype)1., weight_h_diff + 2*H_*H_);
      } else {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, H_, B_, (Dtype)1.,
          temp_bh_.mutable_gpu_data(), prev_out_.gpu_data(),
          (Dtype)1., weight_h_diff + 2*H_*H_);
      }

      caffe_gpu_mul<Dtype>(B_*H_, pre_dg_t,
          cont_values_.mutable_gpu_data(), temp_bh_.mutable_gpu_data());
      if (t > 0) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, H_, B_, (Dtype)1.,
          temp_bh_.mutable_gpu_data(), top_data + (t-1)*B_*H_,
          (Dtype)1., weight_h_diff + 3*H_*H_);
      } else {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, H_, B_, (Dtype)1.,
          temp_bh_.mutable_gpu_data(), prev_out_.gpu_data(),
          (Dtype)1., weight_h_diff + 3*H_*H_);
      }
    }

  }
 
  if (this->param_propagate_down_[0]) {
    // Gradient w.r.t. input-to-hidden weight
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, I_, B_*T_, (Dtype)1.,
        pre_gate_diff_i, bottom_data, (Dtype)0., weight_i_diff);
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, I_, B_*T_, (Dtype)1.,
        pre_gate_diff_f, bottom_data, (Dtype)0., weight_i_diff + H_*I_);
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, I_, B_*T_, (Dtype)1.,
        pre_gate_diff_o, bottom_data, (Dtype)0., weight_i_diff + 2*H_*I_);
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, I_, B_*T_, (Dtype)1.,
        pre_gate_diff_g, bottom_data, (Dtype)0., weight_i_diff + 3*H_*I_);

    //caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 4*H_, I_, T_, (Dtype)1.,
    //    pre_gate_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_gpu_diff());
  }

  if (bias_term_ && this->param_propagate_down_[2]) { 
    // Gradient w.r.t. bias
    caffe_gpu_gemv<Dtype>(CblasTrans, B_*T_, H_, (Dtype)1., pre_gate_diff_i,
        bias_multiplier_.gpu_data(), (Dtype)0.,
        bias_diff);
    caffe_gpu_gemv<Dtype>(CblasTrans, B_*T_, H_, (Dtype)1., pre_gate_diff_f,
        bias_multiplier_.gpu_data(), (Dtype)0.,
        bias_diff + H_);
    caffe_gpu_gemv<Dtype>(CblasTrans, B_*T_, H_, (Dtype)1., pre_gate_diff_o,
        bias_multiplier_.gpu_data(), (Dtype)0.,
        bias_diff + 2*H_);
    caffe_gpu_gemv<Dtype>(CblasTrans, B_*T_, H_, (Dtype)1., pre_gate_diff_g,
        bias_multiplier_.gpu_data(), (Dtype)0.,
        bias_diff + 3*H_);
    //caffe_gpu_gemv<Dtype>(CblasTrans, T_, 4*H_, (Dtype)1., pre_gate_diff,
    //    bias_multiplier_.gpu_data(), (Dtype)0.,
    //    this->blobs_[2]->mutable_gpu_diff());
  }
  if (propagate_down) {
    // Gradient w.r.t. bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, T_*B_, I_, H_, (Dtype)1.,
        pre_gate_diff_i, weight_i, (Dtype)0., bottom_mutable_diff_data);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, T_*B_, I_, H_, (Dtype)1.,
        pre_gate_diff_f, weight_i + H_*I_, (Dtype)1., bottom_mutable_diff_data);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, T_*B_, I_, H_, (Dtype)1.,
        pre_gate_diff_o, weight_i + 2*H_*I_, (Dtype)1., bottom_mutable_diff_data);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, T_*B_, I_, H_, (Dtype)1.,
        pre_gate_diff_g, weight_i + 3*H_*I_, (Dtype)1., bottom_mutable_diff_data);

    //caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, T_, I_, 4*H_, (Dtype)1.,
    //    pre_gate_diff, weight_i, (Dtype)0., (*bottom)[0]->mutable_gpu_diff());
  }


}

template <typename Dtype>
void LstmLinearLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_cont = bottom[1]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const Dtype* weight_i = this->blobs_[0]->gpu_data();
  const Dtype* weight_h = this->blobs_[1]->gpu_data();

  if (bias_term_) {
   const Dtype* bias = this->blobs_[2]->gpu_data();
   Forward_gpu(bottom_data, bottom_cont, top_data, weight_i, weight_h, bias);
  } else {
   Dtype* bias = NULL;
   Forward_gpu(bottom_data, bottom_cont, top_data, weight_i, weight_h, bias);
  }

 }

template <typename Dtype>
void LstmLinearLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {

  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  const Dtype* bottom_cont = (*bottom)[1]->gpu_data();
  Dtype* bottom_mutable_diff_data = (*bottom)[0]->mutable_gpu_diff();
  Dtype* top_diff = top[0]->mutable_gpu_diff();

  const Dtype* weight_i = this->blobs_[0]->gpu_data();
  const Dtype* weight_h = this->blobs_[1]->gpu_data();

  Dtype* weight_i_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* weight_h_diff = this->blobs_[1]->mutable_gpu_diff();

  if (bias_term_) {
   Dtype* bias_diff = this->blobs_[2]->mutable_gpu_diff();
   Backward_gpu(top_data, top_diff, propagate_down[0],
      bottom_data, bottom_cont, bottom_mutable_diff_data,
      weight_i, weight_h, weight_i_diff, weight_h_diff, bias_diff);
  } else {
   Dtype* bias_diff = NULL; 
   Backward_gpu(top_data, top_diff, propagate_down[0],
      bottom_data, bottom_cont, bottom_mutable_diff_data,
      weight_i, weight_h, weight_i_diff, weight_h_diff, bias_diff);
 
  }
}

INSTANTIATE_CLASS(LstmLinearLayer);

}  // namespace caffe

