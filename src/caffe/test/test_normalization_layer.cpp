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
class NormalizationLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  NormalizationLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~NormalizationLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(NormalizationLayerTest, TestDtypesAndDevices);

TYPED_TEST(NormalizationLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  NormalizationLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int i = 0; i < num; ++i) {
    Dtype sum = 0;
    Dtype orig_bottom = 0;
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          Dtype orig_data = this->blob_bottom_->data_at(i,j,k,l);
          sum += data*data;
          orig_bottom += orig_data * orig_data;
        }
      }
    }

    //sum /= channels * height * width;

    const Dtype kErrorBound = 0.001;
    // expect zero norm
    std::cout << "Orig norm: " << orig_bottom;
    EXPECT_NEAR(1, sum, kErrorBound);

  }
}


TYPED_TEST(NormalizationLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  NormalizationLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

}  // namespace caffe
