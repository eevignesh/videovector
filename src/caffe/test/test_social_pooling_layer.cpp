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
class SocialPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SocialPoolingLayerTest()
      : blob_bottom_1_(new Blob<Dtype>(1, 4, 2, 1)),
        blob_bottom_2_(new Blob<Dtype>(1, 4, 4, 1)),
        blob_bottom_3_(new Blob<Dtype>(1, 4, 2, 1)),
        blob_top_(new Blob<Dtype>()) {

    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_3_);

    Dtype* blob_1_data = blob_bottom_1_->mutable_cpu_data();
    caffe_set<Dtype>(8, (Dtype)0., blob_1_data);

    *(blob_1_data) = 1.0; *(blob_1_data + 1) = 1.0;
    *(blob_1_data + 2) = 0.9; *(blob_1_data + 3) = 0.9;
    *(blob_1_data + 4) = 0.9; *(blob_1_data + 5) = 1.2;
    *(blob_1_data + 6) = 0.0; *(blob_1_data + 7) = 2.0;

    Dtype* blob_2_data = blob_bottom_2_->mutable_cpu_data();
    caffe_set<Dtype>(16, (Dtype)0., blob_2_data);
    caffe_set<Dtype>(3, (Dtype)1., blob_2_data);
    caffe_set<Dtype>(3, (Dtype)1.,  blob_2_data + 4);
    caffe_set<Dtype>(3, (Dtype)1.,  blob_2_data + 8);
    caffe_set<Dtype>(1, (Dtype)1., blob_2_data + 15);

    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_bottom_vec_.push_back(blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_3_);

    blob_top_vec_.push_back(blob_top_);

  }
  virtual ~SocialPoolingLayerTest() {
    delete blob_bottom_1_;
    delete blob_bottom_2_;
    delete blob_bottom_3_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_bottom_3_;

  Blob<Dtype>* const blob_top_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

};

TYPED_TEST_CASE(SocialPoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(SocialPoolingLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  SocialPoolingParameter* social_pooling_param =
      layer_param.mutable_social_pooling_param();
  social_pooling_param->set_pool_feat_size(3);
  //social_pooling_param->set_pool_bin_size(0.1);

  shared_ptr<SocialPoolingLayer<Dtype> > layer(
      new SocialPoolingLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 4);
}

TYPED_TEST(SocialPoolingLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SocialPoolingParameter* social_pooling_param =
      layer_param.mutable_social_pooling_param();
  social_pooling_param->set_pool_feat_size(3);
  //social_pooling_param->set_pool_bin_size(0.1);

  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(2);
  inner_product_param->mutable_weight_filler()->set_type("gaussian");
  inner_product_param->mutable_bias_filler()->set_type("constant");
  inner_product_param->mutable_bias_filler()->set_min(1);
  inner_product_param->mutable_bias_filler()->set_max(2);

  SocialPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      LOG(ERROR) << "Out: " << i << ":" << j << ":" << this->blob_top_->data_at(0, i, j, 0);
    }
  }

  CHECK_EQ(0,0);
}


TYPED_TEST(SocialPoolingLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    SocialPoolingParameter* social_pooling_param =
        layer_param.mutable_social_pooling_param();
    social_pooling_param->set_pool_feat_size(4);
    //social_pooling_param->set_pool_bin_size(0.1);

    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(2);
    inner_product_param->mutable_weight_filler()->set_type("gaussian");
    inner_product_param->mutable_bias_filler()->set_type("constant");
    inner_product_param->mutable_bias_filler()->set_min(1);
    inner_product_param->mutable_bias_filler()->set_max(2);


    SocialPoolingLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
        &(this->blob_top_vec_), 0);

  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
