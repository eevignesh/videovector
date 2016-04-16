#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class MaxMarginLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MaxMarginLossLayerTest()
      : blob_bottom_data_true_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_data_bogus_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_true_);
    filler.Fill(this->blob_bottom_data_bogus_);

    blob_bottom_vec_.push_back(blob_bottom_data_true_);
    blob_bottom_vec_.push_back(blob_bottom_data_bogus_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~MaxMarginLossLayerTest() {
    delete blob_bottom_data_true_;
    delete blob_bottom_data_bogus_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_true_;
  Blob<Dtype>* const blob_bottom_data_bogus_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MaxMarginLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(MaxMarginLossLayerTest, TestForwardL1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MaxMarginLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));

  // Compute the actuall l1 loss and see if it fits
  double l1_loss = 0.0;
  
  for (int i = 0; i < this->blob_bottom_data_true_->num(); ++i) {
    for (int j = 0; j < this->blob_bottom_data_true_->channels(); ++j) {
      double diff = static_cast<double> (this->blob_bottom_data_true_->data_at(i,j,0,0)
          - this->blob_bottom_data_bogus_->data_at(i,j,0,0));
      if (diff < 1) {
        l1_loss += 1 - diff;
      }
    }
  }

  l1_loss /= this->blob_bottom_data_true_->count();

  const Dtype kErrorBound = 0.001;
  // expect zero norm
  std::cout << "Orig l1-loss: " << l1_loss;
  EXPECT_NEAR(l1_loss,this->blob_top_loss_->data_at(0,0,0,0) , kErrorBound);
}

TYPED_TEST(MaxMarginLossLayerTest, TestGradientL1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MaxMarginLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 2e-3, 1701, 1, 0.01);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), 0);
}

TYPED_TEST(MaxMarginLossLayerTest, TestGradientL2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Set norm to L2
  MaxMarginLossParameter* max_margin_loss_param = layer_param.mutable_max_margin_loss_param();
  max_margin_loss_param->set_norm(MaxMarginLossParameter_Norm_L2);
  MaxMarginLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), 0);
}

}  // namespace caffe
