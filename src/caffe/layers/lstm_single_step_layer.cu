#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

 template <typename Dtype>
void LstmSingleStepLayer<Dtype>::Forward_gpu(const Dtype* bottom_data,
    const Dtype* bottom_cell_data,
    const Dtype* bottom_hidden_data,
    Dtype* top_data,
    Dtype* top_cell_data,
    const Dtype* weight_i, const Dtype* weight_h,
    const Dtype* bias){

  //LOG(ERROR) << "Forward";
  Dtype* pre_i_t = pre_gate_i_.mutable_gpu_data();
  Dtype* pre_f_t = pre_gate_f_.mutable_gpu_data();
  Dtype* pre_o_t = pre_gate_o_.mutable_gpu_data();
  Dtype* pre_g_t = pre_gate_g_.mutable_gpu_data();

  Dtype* i_t = gate_i_.mutable_gpu_data();
  Dtype* f_t = gate_f_.mutable_gpu_data();
  Dtype* o_t = gate_o_.mutable_gpu_data();
  Dtype* g_t = gate_g_.mutable_gpu_data();

  Dtype* h_t = top_data;
  Dtype* c_t = top_cell_data;
  Dtype* tanh_c_t = tanh_cell_.mutable_gpu_data();

  Dtype* ig = ig_.mutable_gpu_data();

  // Compute input to hidden forward propagation
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, B_*T_, H_, I_, (Dtype)1.,
      bottom_data, weight_i, (Dtype)0., pre_i_t);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, B_*T_, H_, I_, (Dtype)1.,
      bottom_data, weight_i + H_*I_, (Dtype)0., pre_f_t);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, B_*T_, H_, I_, (Dtype)1.,
      bottom_data, weight_i + 2*H_*I_, (Dtype)0., pre_o_t);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, B_*T_, H_, I_, (Dtype)1.,
      bottom_data, weight_i + 3*H_*I_, (Dtype)0., pre_g_t);

  // Add bias 
  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_*T_, H_, 1, (Dtype)1.,
        bias_multiplier_.gpu_data(), bias, (Dtype)1., pre_i_t);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_*T_, H_, 1, (Dtype)1.,
        bias_multiplier_.gpu_data(), bias + H_, (Dtype)1., pre_f_t);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_*T_, H_, 1, (Dtype)1.,
        bias_multiplier_.gpu_data(), bias + 2*H_, (Dtype)1., pre_o_t);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_*T_, H_, 1, (Dtype)1.,
        bias_multiplier_.gpu_data(), bias + 3*H_, (Dtype)1., pre_g_t);

  }
  
  // Add hidden-to-hidden propagation
  const Dtype* h_t_1 = bottom_hidden_data;

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, B_, H_, H_, (Dtype)1.,
      h_t_1, weight_h, (Dtype)1., pre_i_t);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, B_, H_, H_, (Dtype)1.,
      h_t_1, weight_h + H_*H_, (Dtype)1., pre_f_t);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, B_, H_, H_, (Dtype)1.,
      h_t_1, weight_h + 2*H_*H_, (Dtype)1., pre_o_t);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, B_, H_, H_, (Dtype)1.,
      h_t_1, weight_h + 3*H_*H_, (Dtype)1., pre_g_t);

  // Apply nonlinearity
  // Sigmoid - input/forget/output gate
  // TanH - modulation gate
  sigmoid_->Forward_gpu(H_*B_, pre_i_t, i_t);
  sigmoid_->Forward_gpu(H_*B_, pre_f_t, f_t);
  sigmoid_->Forward_gpu(H_*B_, pre_o_t, o_t);

  tanh_->Forward_gpu(H_*B_, pre_g_t, g_t);

  // Compute cell : c(t) = f(t)*c(t-1) + i(t)*g(t)
  caffe_gpu_mul<Dtype>(B_*H_, f_t, bottom_cell_data, c_t);
  caffe_gpu_mul<Dtype>(B_*H_, i_t, g_t, ig);
  caffe_gpu_add<Dtype>(B_*H_, c_t, ig, c_t);

  // Compute output 
  tanh_->Forward_gpu(B_*H_, c_t, tanh_c_t);
  caffe_gpu_mul<Dtype>(B_*H_, o_t, tanh_c_t, h_t);

}

template <typename Dtype>
void LstmSingleStepLayer<Dtype>::Backward_gpu(const Dtype* top_data,
    const Dtype* top_cell_data,
    Dtype* top_diff,
    Dtype* top_cell_diff,
    const vector<bool>& propagate_down,
    const Dtype* bottom_data,
    const Dtype* bottom_cell_data,
    const Dtype* bottom_hidden_data,
    Dtype* bottom_mutable_diff_data,
    Dtype* bottom_mutable_diff_cell_data,
    Dtype* bottom_mutable_diff_hidden_data,
    const Dtype* weight_i, const Dtype* weight_h,
    Dtype* weight_i_diff, Dtype* weight_h_diff, Dtype* bias_diff) {

  Dtype* tanh_c_t = tanh_cell_.mutable_gpu_data();
  Dtype* cell_diff = cell_.mutable_gpu_diff();
  
  caffe_gpu_set<Dtype>(4*H_*H_, (Dtype)0., weight_h_diff);

  const Dtype* i_t = gate_i_.gpu_data();
  const Dtype* f_t = gate_f_.gpu_data(); 
  const Dtype* o_t = gate_o_.gpu_data(); 
  const Dtype* g_t = gate_g_.gpu_data();

  Dtype* di_t = gate_i_.mutable_gpu_diff();
  Dtype* df_t = gate_f_.mutable_gpu_diff(); 
  Dtype* do_t = gate_o_.mutable_gpu_diff(); 
  Dtype* dg_t = gate_g_.mutable_gpu_diff(); 

  Dtype* pre_di_t = pre_gate_i_.mutable_gpu_diff();
  Dtype* pre_df_t = pre_gate_f_.mutable_gpu_diff();
  Dtype* pre_do_t = pre_gate_o_.mutable_gpu_diff();
  Dtype* pre_dg_t = pre_gate_g_.mutable_gpu_diff();

  // Output gate : tanh(c(t)) * h_diff(t)
  caffe_gpu_mul<Dtype>(B_*H_, tanh_c_t, top_diff, do_t);

  // Cell state : o(t) * tanh'(c(t)) * h_diff(t) + f(t+1) * c_diff(t+1)
  caffe_gpu_mul<Dtype>(B_*H_, o_t, top_diff, cell_diff);
  tanh_->Backward_gpu(B_*H_, tanh_c_t, cell_diff, cell_diff);
  caffe_gpu_add<Dtype>(B_*H_, cell_diff, top_cell_diff, cell_diff);
 
  // TODO: include start_sequence values here
  if (bottom_mutable_diff_cell_data) {
    if (propagate_down[1]) {
      caffe_gpu_mul<Dtype>(B_*H_, f_t, cell_diff, bottom_mutable_diff_cell_data);
    }
  }

  // Forget gate : c(t-1) * c_diff(t)  
  caffe_gpu_mul<Dtype>(B_*H_, bottom_cell_data, cell_diff, df_t);

  // Input gate : g(t) * c_diff(t)
  caffe_gpu_mul<Dtype>(B_*H_, g_t, cell_diff, di_t);

  // Input modulation gate : i(t) * c_diff(t)
  caffe_gpu_mul<Dtype>(B_*H_, i_t, cell_diff, dg_t);

  // Compute derivate before nonlinearity
  sigmoid_->Backward_gpu(B_*H_, i_t, di_t, pre_di_t);
  sigmoid_->Backward_gpu(B_*H_, f_t, df_t, pre_df_t);
  sigmoid_->Backward_gpu(B_*H_, o_t, do_t, pre_do_t);

  tanh_->Backward_gpu(B_*H_, g_t, dg_t, pre_dg_t);

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

  if (bottom_mutable_diff_hidden_data) {
    if (propagate_down[2]) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, H_, H_, (Dtype)1.,
          pre_di_t, weight_h, (Dtype)0., bottom_mutable_diff_hidden_data);
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, H_, H_, (Dtype)1.,
          pre_df_t, weight_h + H_*H_, (Dtype)1., bottom_mutable_diff_hidden_data);
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, H_, H_, (Dtype)1.,
          pre_do_t, weight_h + 2*H_*H_, (Dtype)1., bottom_mutable_diff_hidden_data);
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, H_, H_, (Dtype)1.,
          pre_dg_t, weight_h + 3*H_*H_, (Dtype)1., bottom_mutable_diff_hidden_data);
    }
  }

  if (this->param_propagate_down_[1]) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, H_, B_, (Dtype)1.,
      pre_di_t, bottom_hidden_data,
      (Dtype)1., weight_h_diff);
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, H_, B_, (Dtype)1.,
      pre_df_t, bottom_hidden_data,
      (Dtype)1., weight_h_diff + H_*H_);
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, H_, B_, (Dtype)1.,
      pre_do_t, bottom_hidden_data,
      (Dtype)1., weight_h_diff + 2*H_*H_);
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, H_, B_, (Dtype)1.,
      pre_dg_t, bottom_hidden_data,
      (Dtype)1., weight_h_diff + 3*H_*H_);
  }

  if (this->param_propagate_down_[0]) {
    // Gradient w.r.t. input-to-hidden weight
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, I_, B_, (Dtype)1.,
        pre_di_t, bottom_data, (Dtype)0., weight_i_diff);
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, I_, B_*T_, (Dtype)1.,
        pre_df_t, bottom_data, (Dtype)0., weight_i_diff + H_*I_);
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, I_, B_*T_, (Dtype)1.,
        pre_do_t, bottom_data, (Dtype)0., weight_i_diff + 2*H_*I_);
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, I_, B_*T_, (Dtype)1.,
        pre_dg_t, bottom_data, (Dtype)0., weight_i_diff + 3*H_*I_);

    //caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 4*H_, I_, T_, (Dtype)1.,
    //    pre_gate_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_gpu_diff());
  }

  if (bias_term_ && this->param_propagate_down_[2]) { 
    // Gradient w.r.t. bias
    caffe_gpu_gemv<Dtype>(CblasTrans, B_, H_, (Dtype)1., pre_di_t,
        bias_multiplier_.gpu_data(), (Dtype)0., bias_diff);
    caffe_gpu_gemv<Dtype>(CblasTrans, B_, H_, (Dtype)1., pre_df_t,
        bias_multiplier_.gpu_data(), (Dtype)0., bias_diff + H_);
    caffe_gpu_gemv<Dtype>(CblasTrans, B_, H_, (Dtype)1., pre_do_t,
        bias_multiplier_.gpu_data(), (Dtype)0., bias_diff + 2*H_);
    caffe_gpu_gemv<Dtype>(CblasTrans, B_*T_, H_, (Dtype)1., pre_dg_t,
        bias_multiplier_.gpu_data(), (Dtype)0., bias_diff + 3*H_);
  }

  if (propagate_down[0]) {
    // Gradient w.r.t. bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, T_*B_, I_, H_, (Dtype)1.,
        pre_di_t, weight_i, (Dtype)0., bottom_mutable_diff_data);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, T_*B_, I_, H_, (Dtype)1.,
        pre_df_t, weight_i + H_*I_, (Dtype)1., bottom_mutable_diff_data);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, T_*B_, I_, H_, (Dtype)1.,
        pre_do_t, weight_i + 2*H_*I_, (Dtype)1., bottom_mutable_diff_data);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, T_*B_, I_, H_, (Dtype)1.,
        pre_dg_t, weight_i + 3*H_*I_, (Dtype)1., bottom_mutable_diff_data);
  }


}

template <typename Dtype>
void LstmSingleStepLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();

  const Dtype* bottom_cell_data = cell_.gpu_data();
  const Dtype* bottom_hidden_data = cell_.gpu_data();

  if (bottom.size() > 1) {
    bottom_cell_data = bottom[1]->gpu_data();
    bottom_hidden_data = bottom[2]->gpu_data();
  }

  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  Dtype* top_cell_data = (*top)[1]->mutable_gpu_data();

  const Dtype* weight_i = this->blobs_[0]->gpu_data();
  const Dtype* weight_h = this->blobs_[1]->gpu_data();

  const Dtype* bias = this->blobs_[2]->gpu_data();

  if (bias_term_) {
   bias = this->blobs_[2]->gpu_data();
  }
  Forward_gpu(bottom_data, bottom_cell_data, bottom_hidden_data,
      top_data, top_cell_data, weight_i, weight_h, bias);

}

template <typename Dtype>
void LstmSingleStepLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {

  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_cell_data = top[1]->gpu_data();

  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  
  const Dtype* bottom_cell_data = cell_.gpu_data();
  const Dtype* bottom_hidden_data = cell_.gpu_data();

  if (bottom->size() > 1) {
    bottom_cell_data = (*bottom)[1]->gpu_data();
    bottom_hidden_data = (*bottom)[2]->gpu_data();
  }

  Dtype* bottom_mutable_diff_data = (*bottom)[0]->mutable_gpu_diff();
  Dtype* bottom_mutable_diff_cell_data = NULL; 
  Dtype* bottom_mutable_diff_hidden_data = NULL;

  if (bottom->size() > 1) {
    bottom_mutable_diff_cell_data = (*bottom)[1]->mutable_gpu_diff();
    bottom_mutable_diff_hidden_data = (*bottom)[2]->mutable_gpu_diff();
  }

  Dtype* top_diff = top[0]->mutable_gpu_diff();
  Dtype* top_cell_diff = top[1]->mutable_gpu_diff();

  const Dtype* weight_i = this->blobs_[0]->gpu_data();
  const Dtype* weight_h = this->blobs_[1]->gpu_data();

  Dtype* weight_i_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* weight_h_diff = this->blobs_[1]->mutable_gpu_diff();

  Dtype* bias_diff = NULL; 

  if (bias_term_) {
   bias_diff = this->blobs_[2]->mutable_gpu_diff();
  }
  
  Backward_gpu(top_data, top_cell_data, top_diff, top_cell_diff,
       propagate_down, bottom_data, bottom_cell_data, bottom_hidden_data,
       bottom_mutable_diff_data, bottom_mutable_diff_cell_data, bottom_mutable_diff_hidden_data,
       weight_i, weight_h, weight_i_diff, weight_h_diff, bias_diff);
}

 
  
INSTANTIATE_CLASS(LstmSingleStepLayer);

}  // namespace caffe

