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
class RetrievalStatsLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  RetrievalStatsLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(5, 2, 1, 1)),
        blob_bottom_video_ids_(new Blob<Dtype>(5, 1, 1, 1)),
        blob_top_map_(new Blob<Dtype>()),
        blob_top_acc1_(new Blob<Dtype>()),
        blob_top_acc5_(new Blob<Dtype>()) {

    // fill the values
    Dtype* bottom_data = blob_bottom_data_->mutable_cpu_data();
    Dtype* bottom_video_ids = blob_bottom_video_ids_->mutable_cpu_data();

    Dtype feat_data[] = {1.0, 0.0,
                         0.0, 1.0,
                         1.0, 0.06,
                         0.0, 1.0,
                         1.0, 0.1};
    Dtype video_ids[] = {2, 3, 4, 5, 6}; // 1,2,1,2,2
    
    caffe_copy<Dtype>(10, feat_data, bottom_data);
    caffe_copy<Dtype>(5, video_ids, bottom_video_ids);

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_video_ids_);

    blob_top_vec_.push_back(blob_top_map_);
    blob_top_vec_.push_back(blob_top_acc1_);
    blob_top_vec_.push_back(blob_top_acc5_);
  }
  virtual ~RetrievalStatsLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_video_ids_;
    delete blob_top_map_;
    delete blob_top_acc1_;
    delete blob_top_acc5_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_video_ids_;

  Blob<Dtype>* const blob_top_map_;
  Blob<Dtype>* const blob_top_acc1_;
  Blob<Dtype>* const blob_top_acc5_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(RetrievalStatsLayerTest, TestDtypesAndDevices);

TYPED_TEST(RetrievalStatsLayerTest, TestForwardL1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  auto retpar = layer_param.mutable_retrieval_stats_param();
  retpar->set_id_to_class_file("/afs/cs.stanford.edu/u/vigneshr/scratch/ICCV2015/data/_temp/dummy.txt");
  RetrievalStatsLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));

  const Dtype kErrorBound = 0.001;

  EXPECT_NEAR(0.7833333, this->blob_top_map_->data_at(0,0,0,0), kErrorBound);
  EXPECT_NEAR(0.60, this->blob_top_acc1_->data_at(0,0,0,0), kErrorBound);
  EXPECT_NEAR(0.32, this->blob_top_acc5_->data_at(0,0,0,0), kErrorBound);

}


}  // namespace caffe
