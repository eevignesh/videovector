#ifndef CAFFE_COMMON_LAYERS_HPP_
#define CAFFE_COMMON_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Compute the index of the @f$ K @f$ max values for each datum across
 *        all dimensions @f$ (C \times H \times W) @f$.
 *
 * Intended for use after a classification layer to produce a prediction.
 * If parameter out_max_val is set to true, output is a vector of pairs
 * (max_ind, max_val) for each image.
 *
 * NOTE: does not implement Backwards operation.
 */
template <typename Dtype>
class ArgMaxLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides ArgMaxParameter argmax_param,
   *     with ArgMaxLayer options:
   *   - top_k (\b optional uint, default 1).
   *     the number @f$ K @f$ of maximal items to output.
   *   - out_max_val (\b optional bool, default false).
   *     if set, output a vector of pairs (max_ind, max_val) for each image.
   */
  explicit ArgMaxLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_ARGMAX;
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times 1 \times K \times 1) @f$ or, if out_max_val
   *      @f$ (N \times 2 \times K \times 1) @f$
   *      the computed outputs @f$
   *       y_n = \arg\max\limits_i x_{ni}
   *      @f$ (for @f$ K = 1 @f$).
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
    NOT_IMPLEMENTED;
  }
  bool out_max_val_;
  size_t top_k_;
};

/**
 * @brief Takes at least two Blob%s and concatenates them along either the num
 *        or channel dimension, outputting the result.
 */
template <typename Dtype>
class ConcatLayer : public Layer<Dtype> {
 public:
  explicit ConcatLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_CONCAT;
  }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @param bottom input Blob vector (length 2+)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x_1 @f$
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x_2 @f$
   *   -# ...
   *   - K @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x_K @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (KN \times C \times H \times W) @f$ if concat_dim == 0, or
   *      @f$ (N \times KC \times H \times W) @f$ if concat_dim == 1:
   *      the concatenated output @f$
   *        y = [\begin{array}{cccc} x_1 & x_2 & ... & x_K \end{array}]
   *      @f$
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  /**
   * @brief Computes the error gradient w.r.t. the concatenate inputs.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *        respect to the outputs
   *   -# @f$ (KN \times C \times H \times W) @f$ if concat_dim == 0, or
   *      @f$ (N \times KC \times H \times W) @f$ if concat_dim == 1:
   *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
   *      with respect to concatenated outputs @f$ y @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length K), into which the top gradient
   *        @f$ \frac{\partial E}{\partial y} @f$ is deconcatenated back to the
   *        inputs @f$
   *        \left[ \begin{array}{cccc}
   *          \frac{\partial E}{\partial x_1} &
   *          \frac{\partial E}{\partial x_2} &
   *          ... &
   *          \frac{\partial E}{\partial x_K}
   *        \end{array} \right] =
   *        \frac{\partial E}{\partial y}
   *        @f$
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  Blob<Dtype> col_bob_;
  int count_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int concat_dim_;
};

/**
 * @brief Compute elementwise operations, such as product and sum,
 *        along multiple input Blobs.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class EltwiseLayer : public Layer<Dtype> {
 public:
  explicit EltwiseLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_ELTWISE;
  }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  EltwiseParameter_EltwiseOp op_;
  vector<Dtype> coeffs_;
  Blob<int> max_idx_;

  bool stable_prod_grad_;
};

/**
 * @brief Reshapes the input Blob into flat vectors.
 *
 * Note: because this layer does not change the input values -- merely the
 * dimensions -- it can simply copy the input. The copy happens "virtually"
 * (thus taking effectively 0 real time) by setting, in Forward, the data
 * pointer of the top Blob to that of the bottom Blob (see Blob::ShareData),
 * and in Backward, the diff pointer of the bottom Blob to that of the top Blob
 * (see Blob::ShareDiff).
 */
template <typename Dtype>
class FlattenBatchLayer : public Layer<Dtype> {
 public:
  explicit FlattenBatchLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_FLATTEN_BATCH;
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @param bottom input Blob vector (length 2+)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times CHW \times 1 \times 1) @f$
   *      the outputs -- i.e., the (virtually) copied, flattened inputs
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  /**
   * @brief Computes the error gradient w.r.t. the concatenate inputs.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *        respect to the outputs
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length K), into which the top error
   *        gradient is (virtually) copied
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  int count_;
};

/**
 * @brief Reshapes the input Blob into flat vectors.
 *
 * Note: because this layer does not change the input values -- merely the
 * dimensions -- it can simply copy the input. The copy happens "virtually"
 * (thus taking effectively 0 real time) by setting, in Forward, the data
 * pointer of the top Blob to that of the bottom Blob (see Blob::ShareData),
 * and in Backward, the diff pointer of the bottom Blob to that of the top Blob
 * (see Blob::ShareDiff).
 */
template <typename Dtype>
class FlattenLayer : public Layer<Dtype> {
 public:
  explicit FlattenLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_FLATTEN;
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @param bottom input Blob vector (length 2+)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times CHW \times 1 \times 1) @f$
   *      the outputs -- i.e., the (virtually) copied, flattened inputs
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  /**
   * @brief Computes the error gradient w.r.t. the concatenate inputs.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *        respect to the outputs
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length K), into which the top error
   *        gradient is (virtually) copied
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  int count_;
};

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class InnerProductLayer : public Layer<Dtype> {
 public:
  explicit InnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_INNER_PRODUCT;
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
};

/**
 * @brief Fetches the weight vectors corresponding to the indices provided in bottom data.
 *        The weight vectors are learned, however the indices are constant.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class IdToWeightMappingLayer : public Layer<Dtype> {
 public:
  explicit IdToWeightMappingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_ID_TO_WEIGHT_MAPPING;
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  int M_;
  int K_;
  int N_;
};


/**
 * @brief Normalizes the input to norm of 1.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class NormalizationLayer : public Layer<Dtype> {
 public:
  explicit NormalizationLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_MVN;
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  Blob<Dtype> sum_square_, temp_, temp2_;

  /// sum_multiplier is used to carry out sum using BLAS
  Blob<Dtype> sum_multiplier_;
};


/**
 * @brief Normalizes the input to have 0-mean and/or unit (1) variance.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class MVNLayer : public Layer<Dtype> {
 public:
  explicit MVNLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_MVN;
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  Blob<Dtype> mean_, variance_, temp_;

  /// sum_multiplier is used to carry out sum using BLAS
  Blob<Dtype> sum_multiplier_;
};

/**
 * @brief Ignores bottom blobs while producing no top blobs. (This is useful
 *        to suppress outputs during testing.)
 */
template <typename Dtype>
class SilenceLayer : public Layer<Dtype> {
 public:
  explicit SilenceLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {}

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_SILENCE;
  }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {}
  // We can't define Forward_gpu here, since STUB_GPU will provide
  // its own definition for CPU_ONLY mode.
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
};

/**
 * @brief Computes the softmax function.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_SOFTMAX;
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  /// sum_multiplier is used to carry out sum using BLAS
  Blob<Dtype> sum_multiplier_;
  /// scale is an intermediate Blob to hold temporary results.
  Blob<Dtype> scale_;
};

#ifdef USE_CUDNN
/**
 * @brief cuDNN implementation of SoftmaxLayer.
 *        Fallback to SoftmaxLayer for CPU mode.
 */
template <typename Dtype>
class CuDNNSoftmaxLayer : public SoftmaxLayer<Dtype> {
 public:
  explicit CuDNNSoftmaxLayer(const LayerParameter& param)
      : SoftmaxLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual ~CuDNNSoftmaxLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  cudnnHandle_t             handle_;
  cudnnTensor4dDescriptor_t bottom_desc_;
  cudnnTensor4dDescriptor_t top_desc_;
};
#endif

/**
 * @brief Creates a "split" path in the network by copying the bottom Blob
 *        into multiple top Blob%s to be used by multiple consuming layers.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SplitLayer : public Layer<Dtype> {
 public:
  explicit SplitLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_SPLIT;
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  int count_;
};

/**
 * @brief Takes a Blob and slices it along either the num or channel dimension,
 *        outputting multiple sliced Blob results.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SliceLayer : public Layer<Dtype> {
 public:
  explicit SliceLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_SLICE;
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  Blob<Dtype> col_bob_;
  int count_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int slice_dim_;
  vector<int> slice_point_;
};

/**
 * @brief Sums the values across all channels, width, height. If sum_layer_param().num_output()
 *  > 1, then it replicates it along different channels.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SumLayer : public Layer<Dtype> {
 public:
  explicit SumLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_SUM;
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  Blob<Dtype> temp_;
  int num_output_;
  /// sum_multiplier is used to carry out sum using BLAS
  Blob<Dtype> sum_multiplier_;
  Blob<Dtype> sum_multiplier_2_;
};

/**
 * @brief Long-short term memory cell layer, computes an inner product
 *        with a set of learned weights with recurrent connections,
 *        and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class LstmLayer : public Layer<Dtype> {
 public:
  explicit LstmLayer(const LayerParameter& param)
      : Layer<Dtype>(param), sigmoid_(new SigmoidLayer<Dtype>(param)),
       tanh_(new TanHLayer<Dtype>(param)), encoder_mode_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_LSTM;
  }

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual void LayerSetUp(Blob<Dtype>* const& bottom_data);
  virtual void Reshape(const Blob<Dtype>* bottom_data,
      const Blob<Dtype>* bottom_cont, Blob<Dtype>* top);

  virtual void Forward_cpu(const Dtype* bottom_data,
    const Dtype* bottom_cont, Dtype* top_data,
    const Dtype* weight_i, const Dtype* weight_h, const Dtype* bias);

  virtual void Backward_cpu(const Dtype* top_data, Dtype* top_diff,
    const bool& propogate_down, const Dtype* bottom_data,
    const Dtype* bottom_cont, Dtype* bottom_mutable_diff_data,
    const Dtype* weight_i, const Dtype* weight_h,
    Dtype* weight_i_diff, Dtype* weight_h_diff, Dtype* bias_diff);

  virtual void Forward_gpu(const Dtype* bottom_data,
    const Dtype* bottom_cont, Dtype* top_data,
    const Dtype* weight_i, const Dtype* weight_h, const Dtype* bias);

  virtual void Backward_gpu(const Dtype* top_data, Dtype* top_diff,
    const bool& propogate_down, const Dtype* bottom_data,
    const Dtype* bottom_cont, Dtype* bottom_mutable_diff_data,
    const Dtype* weight_i, const Dtype* weight_h,
    Dtype* weight_i_diff, Dtype* weight_h_diff, Dtype* bias_diff);

  Blob<Dtype> next_cell_; // next cell state value
  Blob<Dtype> next_out_;  // next hidden activation value
  Blob<Dtype> next_cell_diff_; // the gradient of the cell from next time-step

  virtual inline void SetAsEncoder(const bool isEncoder) {
    encoder_mode_ = isEncoder;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  int I_; // input dimension
  int H_; // num of hidden units
  int T_; // length of sequence
  int B_; // batch size

  bool bias_term_;
  bool encoder_mode_;

  Blob<Dtype> bias_multiplier_;
  Blob<Dtype> cont_multiplier_;
  Blob<Dtype> cont_values_;

  Dtype clipping_threshold_; // threshold for clipped gradient
  Blob<Dtype> pre_gate_i_;  // gate values before nonlinearity
  Blob<Dtype> pre_gate_f_;  // gate values before nonlinearity
  Blob<Dtype> pre_gate_o_;  // gate values before nonlinearity
  Blob<Dtype> pre_gate_g_;  // gate values before nonlinearity

  Blob<Dtype> gate_i_;      // gate values after nonlinearity
  Blob<Dtype> gate_f_;      // gate values after nonlinearity
  Blob<Dtype> gate_o_;      // gate values after nonlinearity
  Blob<Dtype> gate_g_;      // gate values after nonlinearity

  Blob<Dtype> cell_;      // memory cell;

  Blob<Dtype> prev_cell_; // previous cell state value
  Blob<Dtype> prev_out_;  // previous hidden activation value

  // intermediate values
  Blob<Dtype> fdc_;
  Blob<Dtype> ig_;
  Blob<Dtype> tanh_cell_;
  Blob<Dtype> temp_bh_;

  shared_ptr<SigmoidLayer<Dtype> > sigmoid_;
  shared_ptr<TanHLayer<Dtype> > tanh_;
};

/**
 * @brief Long-short term memory cell layer, computes an inner product
 *        with a set of learned weights with recurrent connections,
 *        and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class LstmEncDecLayer : public Layer<Dtype> {
 public:
  explicit LstmEncDecLayer(const LayerParameter& param)
      : Layer<Dtype>(param), encoder_lstm_(new LstmLayer<Dtype>(param)),
      decoder_lstm_(new LstmLayer<Dtype>(param)) {}
  
  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_LSTM_ENC_DEC;
  }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline int ExactNumBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  int H_; // num of hidden units
  int T_; // length of sequence
  int B_; // batch size
  int I_; // feature dimension
  bool bias_term_;

  shared_ptr<LstmLayer<Dtype> > encoder_lstm_;
  shared_ptr<LstmLayer<Dtype> > decoder_lstm_;

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

};

template <typename Dtype>
class LstmConditionalLayer : public Layer<Dtype> {
 public:
  explicit LstmConditionalLayer(const LayerParameter& param)
      : Layer<Dtype>(param), sigmoid_(new SigmoidLayer<Dtype>(param)),
       tanh_(new TanHLayer<Dtype>(param)), encoder_mode_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_LSTM_CONDITIONAL;
  }

  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual void LayerSetUp(Blob<Dtype>* const& bottom_data);
  virtual void Reshape(const Blob<Dtype>* bottom_data,
      const Blob<Dtype>* bottom_cont, Blob<Dtype>* top);

  virtual void Forward_cpu(const Dtype* bottom_data,
    const Dtype* first_input,
    const Dtype* bottom_cont, Dtype* top_output_data,
    const Dtype* weight_i, const Dtype* weight_h,
    const Dtype* weight_h2o, const Dtype* weight_o2h,
    const Dtype* bias, const Dtype* bias_h2o, const Dtype* bias_o2h);

  virtual void Backward_cpu(const Dtype* first_input,
    const Dtype* top_output_data,
    Dtype* top_output_diff,
    const bool& propagate_down, const Dtype* bottom_data,
    const Dtype* bottom_cont, Dtype* bottom_mutable_diff_data,
    const Dtype* weight_i, const Dtype* weight_h,
    const Dtype* weight_h2o, const Dtype* weight_o2h,
    Dtype* weight_i_diff, Dtype* weight_h_diff,
    Dtype* weight_h2o_diff, Dtype* weight_o2h_diff,
    Dtype* bias_diff, Dtype* bias_h2o_diff, Dtype* bias_o2h_diff);

  virtual void Forward_gpu(const Dtype* bottom_data,
    const Dtype* first_input,
    const Dtype* bottom_cont, Dtype* top_output_data,
    const Dtype* weight_i, const Dtype* weight_h,
    const Dtype* weight_h2o, const Dtype* weight_o2h,
    const Dtype* bias, const Dtype* bias_h2o, const Dtype* bias_o2h);

  virtual void Backward_gpu(const Dtype* first_input,
    const Dtype* top_output_data,
    Dtype* top_output_diff,
    const bool& propagate_down, const Dtype* bottom_data,
    const Dtype* bottom_cont, Dtype* bottom_mutable_diff_data,
    const Dtype* weight_i, const Dtype* weight_h,
    const Dtype* weight_h2o, const Dtype* weight_o2h,
    Dtype* weight_i_diff, Dtype* weight_h_diff,
    Dtype* weight_h2o_diff, Dtype* weight_o2h_diff,
    Dtype* bias_diff, Dtype* bias_h2o_diff, Dtype* bias_o2h_diff);

  Blob<Dtype> next_cell_; // next cell state value
  Blob<Dtype> next_out_;  // next hidden activation value
  Blob<Dtype> next_cell_diff_; // the gradient of the cell from next time-step

  virtual inline void SetAsEncoder(const bool isEncoder) {
    encoder_mode_ = isEncoder;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  int I_; // input dimension
  int H_; // num of hidden units
  int T_; // length of sequence
  int B_; // batch size
  int O_; // output size

  bool bias_term_;
  bool encoder_mode_;

  Blob<Dtype> bias_multiplier_;
  Blob<Dtype> cont_multiplier_;
  Blob<Dtype> cont_values_;

  Dtype clipping_threshold_; // threshold for clipped gradient
  Blob<Dtype> pre_gate_i_;  // gate values before nonlinearity
  Blob<Dtype> pre_gate_f_;  // gate values before nonlinearity
  Blob<Dtype> pre_gate_o_;  // gate values before nonlinearity
  Blob<Dtype> pre_gate_g_;  // gate values before nonlinearity

  Blob<Dtype> gate_i_;      // gate values after nonlinearity
  Blob<Dtype> gate_f_;      // gate values after nonlinearity
  Blob<Dtype> gate_o_;      // gate values after nonlinearity
  Blob<Dtype> gate_g_;      // gate values after nonlinearity

  Blob<Dtype> cell_;      // memory cell;

  Blob<Dtype> hidden_data_;

  Blob<Dtype> prev_cell_; // previous cell state value
  Blob<Dtype> prev_out_;  // previous hidden activation value

  // intermediate values
  Blob<Dtype> fdc_;
  Blob<Dtype> ig_;
  Blob<Dtype> tanh_cell_;
  Blob<Dtype> temp_bh_;
  Blob<Dtype> pre_tanh_out_;


  shared_ptr<SigmoidLayer<Dtype> > sigmoid_;
  shared_ptr<TanHLayer<Dtype> > tanh_;
};

template <typename Dtype>
class LstmLinearLayer : public Layer<Dtype> {
 public:
  explicit LstmLinearLayer(const LayerParameter& param)
      : Layer<Dtype>(param), sigmoid_(new SigmoidLayer<Dtype>(param)),
       tanh_(new TanHLayer<Dtype>(param)), encoder_mode_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_LSTM_LINEAR;
  }

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual void LayerSetUp(Blob<Dtype>* const& bottom_data);
  virtual void Reshape(const Blob<Dtype>* bottom_data,
      const Blob<Dtype>* bottom_cont, Blob<Dtype>* top);

  virtual void Forward_cpu(const Dtype* bottom_data,
    const Dtype* bottom_cont, Dtype* top_data,
    const Dtype* weight_i, const Dtype* weight_h, const Dtype* bias);

  virtual void Backward_cpu(const Dtype* top_data, Dtype* top_diff,
    const bool& propogate_down, const Dtype* bottom_data,
    const Dtype* bottom_cont, Dtype* bottom_mutable_diff_data,
    const Dtype* weight_i, const Dtype* weight_h,
    Dtype* weight_i_diff, Dtype* weight_h_diff, Dtype* bias_diff);

  virtual void Forward_gpu(const Dtype* bottom_data,
    const Dtype* bottom_cont, Dtype* top_data,
    const Dtype* weight_i, const Dtype* weight_h, const Dtype* bias);

  virtual void Backward_gpu(const Dtype* top_data, Dtype* top_diff,
    const bool& propogate_down, const Dtype* bottom_data,
    const Dtype* bottom_cont, Dtype* bottom_mutable_diff_data,
    const Dtype* weight_i, const Dtype* weight_h,
    Dtype* weight_i_diff, Dtype* weight_h_diff, Dtype* bias_diff);

  Blob<Dtype> next_cell_; // next cell state value
  Blob<Dtype> next_out_;  // next hidden activation value
  Blob<Dtype> next_cell_diff_; // the gradient of the cell from next time-step

  virtual inline void SetAsEncoder(const bool isEncoder) {
    encoder_mode_ = isEncoder;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  int I_; // input dimension
  int H_; // num of hidden units
  int T_; // length of sequence
  int B_; // batch size

  bool bias_term_;
  bool encoder_mode_;

  Blob<Dtype> bias_multiplier_;
  Blob<Dtype> cont_multiplier_;
  Blob<Dtype> cont_values_;

  Dtype clipping_threshold_; // threshold for clipped gradient
  Blob<Dtype> pre_gate_i_;  // gate values before nonlinearity
  Blob<Dtype> pre_gate_f_;  // gate values before nonlinearity
  Blob<Dtype> pre_gate_o_;  // gate values before nonlinearity
  Blob<Dtype> pre_gate_g_;  // gate values before nonlinearity

  Blob<Dtype> gate_i_;      // gate values after nonlinearity
  Blob<Dtype> gate_f_;      // gate values after nonlinearity
  Blob<Dtype> gate_o_;      // gate values after nonlinearity
  Blob<Dtype> gate_g_;      // gate values after nonlinearity

  Blob<Dtype> cell_;      // memory cell;

  Blob<Dtype> prev_cell_; // previous cell state value
  Blob<Dtype> prev_out_;  // previous hidden activation value

  // intermediate values
  Blob<Dtype> fdc_;
  Blob<Dtype> ig_;
  Blob<Dtype> temp_bh_;

  shared_ptr<SigmoidLayer<Dtype> > sigmoid_;
  shared_ptr<TanHLayer<Dtype> > tanh_;
};

/**
 * @brief Long-short term memory with single cell layer, computes an inner product
 *        with a set of learned weights with recurrent connections,
 *        and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class LstmSingleStepLayer : public Layer<Dtype> {
 public:
  explicit LstmSingleStepLayer(const LayerParameter& param)
      : Layer<Dtype>(param), sigmoid_(new SigmoidLayer<Dtype>(param)),
       tanh_(new TanHLayer<Dtype>(param)){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_LSTM_SINGLE_STEP;
  }

  //virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

  virtual void LayerSetUp(Blob<Dtype>* const& bottom_data);
  virtual void Reshape(const Blob<Dtype>* bottom_data,
      Blob<Dtype>* top, Blob<Dtype>* top_cell);

  virtual void Forward_cpu(const Dtype* bottom_data,
    const Dtype* bottom_cell_data, const Dtype* bottom_hidden_data,
    Dtype* top_data, Dtype* top_cell_data,
    const Dtype* weight_i, const Dtype* weight_h,
    const Dtype* bias);

  virtual void Backward_cpu(const Dtype* top_data,
    const Dtype* top_cell_data, Dtype* top_diff,
    Dtype* top_cell_diff, const vector<bool>& propagate_down,
    const Dtype* bottom_data, const Dtype* bottom_cell_data,
    const Dtype* bottom_hidden_data, Dtype* bottom_mutable_diff_data,
    Dtype* bottom_mutable_diff_cell_data, Dtype* bottom_mutable_diff_hidden_data,
    const Dtype* weight_i, const Dtype* weight_h,
    Dtype* weight_i_diff, Dtype* weight_h_diff, Dtype* bias_diff);

  virtual void Forward_gpu(const Dtype* bottom_data,
    const Dtype* bottom_cell_data, const Dtype* bottom_hidden_data,
    Dtype* top_data, Dtype* top_cell_data,
    const Dtype* weight_i, const Dtype* weight_h,
    const Dtype* bias);

  virtual void Backward_gpu(const Dtype* top_data,
    const Dtype* top_cell_data, Dtype* top_diff,
    Dtype* top_cell_diff, const vector<bool>& propagate_down,
    const Dtype* bottom_data, const Dtype* bottom_cell_data,
    const Dtype* bottom_hidden_data, Dtype* bottom_mutable_diff_data,
    Dtype* bottom_mutable_diff_cell_data, Dtype* bottom_mutable_diff_hidden_data,
    const Dtype* weight_i, const Dtype* weight_h,
    Dtype* weight_i_diff, Dtype* weight_h_diff, Dtype* bias_diff);


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);


  int I_; // input dimension
  int H_; // num of hidden units
  int T_; // length of sequence
  int B_; // batch size

  bool bias_term_;

  Blob<Dtype> bias_multiplier_;

  Dtype clipping_threshold_; // threshold for clipped gradient
  Blob<Dtype> pre_gate_i_;  // gate values before nonlinearity
  Blob<Dtype> pre_gate_f_;  // gate values before nonlinearity
  Blob<Dtype> pre_gate_o_;  // gate values before nonlinearity
  Blob<Dtype> pre_gate_g_;  // gate values before nonlinearity

  Blob<Dtype> gate_i_;      // gate values after nonlinearity
  Blob<Dtype> gate_f_;      // gate values after nonlinearity
  Blob<Dtype> gate_o_;      // gate values after nonlinearity
  Blob<Dtype> gate_g_;      // gate values after nonlinearity

  Blob<Dtype> cell_;      // memory cell;

  Blob<Dtype> ig_;
  Blob<Dtype> tanh_cell_;

  shared_ptr<SigmoidLayer<Dtype> > sigmoid_;
  shared_ptr<TanHLayer<Dtype> > tanh_;
};



}  // namespace caffe

#endif  // CAFFE_COMMON_LAYERS_HPP_
