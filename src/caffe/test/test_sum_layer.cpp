#include <cmath>
#include <cstring>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "gtest/gtest.h"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SumLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SumLayerTest()
      : blob_bottom_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SumLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SumLayerTest, TestDtypesAndDevices);

TYPED_TEST(SumLayerTest, TestForwardNum1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  int num_output = 1;
  SumParameter* sum_param = layer_param.mutable_sum_param();
  sum_param->set_num_output(num_output);
  SumLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  const Dtype kErrorBound = 0.001;

  for (int i = 0; i < num; ++i) {
    Dtype sum = 0;
    for (int j = 0; j < channels; ++j) {
      Dtype orig_data = this->blob_bottom_->data_at(i,j,0,0);
      sum += orig_data;
    }

    for (int j = 0; j < num_output; ++j) {
      Dtype data = this->blob_top_->data_at(i, j, 0, 0);
      EXPECT_NEAR(data, sum, kErrorBound);
    }

  }
}

TYPED_TEST(SumLayerTest, TestForwardNum10) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  int num_output = 10;
  SumParameter* sum_param = layer_param.mutable_sum_param();
  sum_param->set_num_output(num_output);
  SumLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  const Dtype kErrorBound = 0.001;

  for (int i = 0; i < num; ++i) {
    Dtype sum = 0;
    for (int j = 0; j < channels; ++j) {
      Dtype orig_data = this->blob_bottom_->data_at(i,j,0,0);
      sum += orig_data;
    }

    for (int j = 0; j < num_output; ++j) {
      Dtype data = this->blob_top_->data_at(i, j, 0, 0);
      EXPECT_NEAR(data, sum, kErrorBound);
    }

  }
}


TYPED_TEST(SumLayerTest, TestGradientNum1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SumLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(SumLayerTest, TestGradientNum10) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  int num_output = 10;
  SumParameter* sum_param = layer_param.mutable_sum_param();
  sum_param->set_num_output(num_output);
  SumLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

}  // namespace caffe
