#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/common_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class LstmSingleStepLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  LstmSingleStepLayerTest()
      : blob_bottom_1_(new Blob<Dtype>(1, 4, 2, 1)),
        blob_top_1_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()),
        blob_top_3_(new Blob<Dtype>()),
        blob_top_4_(new Blob<Dtype>()) {

    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_1_);

    blob_bottom_vec_.push_back(blob_bottom_1_);

    blob_top_vec_1_.push_back(blob_top_1_);
    blob_top_vec_1_.push_back(blob_top_2_);
    blob_top_vec_2_.push_back(blob_top_3_);
    blob_top_vec_2_.push_back(blob_top_4_);


  }
  virtual ~LstmSingleStepLayerTest() {
    delete blob_bottom_1_;

    delete blob_top_1_;
    delete blob_top_2_;
    delete blob_top_3_;
    delete blob_top_4_;

  }

  Blob<Dtype>* const blob_bottom_1_;

  Blob<Dtype>* const blob_top_1_;
  Blob<Dtype>* const blob_top_2_;
  Blob<Dtype>* const blob_top_3_;
  Blob<Dtype>* const blob_top_4_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_1_;
  vector<Blob<Dtype>*> blob_top_vec_2_;

};

TYPED_TEST_CASE(LstmSingleStepLayerTest, TestDtypesAndDevices);

TYPED_TEST(LstmSingleStepLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(2);
  inner_product_param->mutable_weight_filler()->set_type("gaussian");
  inner_product_param->mutable_bias_filler()->set_type("constant");
  inner_product_param->mutable_weight_filler()->set_std(0.1);

  LstmParameter* lstm_param = layer_param.mutable_lstm_param();
  lstm_param->set_clipping_threshold(1e21);

  shared_ptr<LstmSingleStepLayer<Dtype> > layer(
      new LstmSingleStepLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_1_));
  
  EXPECT_EQ(this->blob_top_1_->num(), 1);
  EXPECT_EQ(this->blob_top_1_->height(), 2);
  EXPECT_EQ(this->blob_top_1_->width(), 1);
  EXPECT_EQ(this->blob_top_1_->channels(), 4);

  EXPECT_EQ(this->blob_top_2_->num(), 1);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->channels(), 4);

}

TYPED_TEST(LstmSingleStepLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(2);
    inner_product_param->mutable_weight_filler()->set_type("gaussian");
    inner_product_param->mutable_bias_filler()->set_type("constant");
    inner_product_param->mutable_bias_filler()->set_min(1);
    inner_product_param->mutable_bias_filler()->set_max(2);

    LstmParameter* lstm_param = layer_param.mutable_lstm_param();
    lstm_param->set_clipping_threshold(1e30);

    LstmSingleStepLayer<Dtype> layer_1(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer_1, &(this->blob_bottom_vec_),
        &(this->blob_top_vec_1_), 0);

    this->blob_bottom_vec_.push_back(this->blob_top_1_);
    this->blob_bottom_vec_.push_back(this->blob_top_2_);
    checker.CheckGradientExhaustive(&layer_1, &(this->blob_bottom_vec_),
        &(this->blob_top_vec_2_), 0);

  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
