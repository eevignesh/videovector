#include <leveldb/db.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/vignesh_util.hpp"
#include "caffe/proto/video_shot_sentences.pb.h"

using video_shot_sentences::TestVideoShotWindows;

namespace caffe {

template <typename Dtype>
VideoShotWindowTestDataLayer<Dtype>::~VideoShotWindowTestDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.video_shot_window_test_data_param().backend()) {
  case VideoShotWindowTestDataParameter_DB_LEVELDB:
    break;  // do nothing
  case VideoShotWindowTestDataParameter_DB_LMDB:
    mdb_cursor_close(mdb_cursor_);
    mdb_close(mdb_env_, mdb_dbi_);
    mdb_txn_abort(mdb_txn_);
    mdb_env_close(mdb_env_);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

template <typename Dtype>
void VideoShotWindowTestDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  // Initialize DB
  switch (this->layer_param_.video_shot_window_test_data_param().backend()) {
  case VideoShotWindowTestDataParameter_DB_LEVELDB:
    {
      leveldb::DB* db_temp;
      leveldb::Options options = GetLevelDBOptions();
      options.create_if_missing = false;
      LOG(INFO) << "Opening leveldb " << this->layer_param_.video_shot_window_test_data_param().source();
      leveldb::Status status = leveldb::DB::Open(
          options, this->layer_param_.video_shot_window_test_data_param().source(), &db_temp);
      CHECK(status.ok()) << "Failed to open leveldb "
                         << this->layer_param_.video_shot_window_test_data_param().source() << std::endl
                         << status.ToString();
      db_.reset(db_temp);
      iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
      iter_->SeekToFirst();
    }
    break;
  case VideoShotWindowTestDataParameter_DB_LMDB:
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.video_shot_window_test_data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.video_shot_window_test_data_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";

    break;
  }

  // Read a data point, and use it to initialize the top blob.
  TestVideoShotWindows test_shot_windows;
  switch (this->layer_param_.video_shot_window_test_data_param().backend()) {
  case VideoShotWindowTestDataParameter_DB_LEVELDB:
    test_shot_windows.ParseFromString(iter_->value().ToString());
    break;
  case VideoShotWindowTestDataParameter_DB_LMDB:
    LOG(INFO) << "Parsing the test shot window";
    test_shot_windows.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }


  // We expect one-d fetures only
  //CHECK_GE(test_shot_windows.positive_shot_words_size(), 1);
  //CHECK_GE(test_shot_windows.negative_shot_words_size(), 1);
  positive_size_ = test_shot_windows.positive_shot_words_size();
  negative_size_ = test_shot_windows.negative_shot_words_size();

  if (!this->layer_param_.video_shot_window_test_data_param().include_positives()) {
    positive_size_ = 0;
  }

  if (!this->layer_param_.video_shot_window_test_data_param().include_negatives()) {
    negative_size_ = 0;
  }

  LOG(INFO) << "Pos-size: " << positive_size_
            << "Neg-size: " << negative_size_;

  feature_size_ = test_shot_windows.context_shot_words(0).float_data_size();
  context_size_ = test_shot_windows.context_shot_words_size();
  CHECK_GE(feature_size_, 1);
  CHECK_GE(context_size_, 1); // need atleast a context of two words
  

  // We use a small hack: channels = context_size_+1, height = feature_size_
  (*top)[0]->Reshape(
      this->layer_param_.video_shot_window_test_data_param().batch_size(),
        context_size_+ positive_size_ + negative_size_,
        feature_size_, 1);

  this->prefetch_data_.Reshape(this->layer_param_.video_shot_window_test_data_param().batch_size(),
      context_size_ + positive_size_ + negative_size_, feature_size_, 1);

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();

  // Video id if needed
  if (this->output_labels_) {
    (*top)[1]->Reshape(this->layer_param_.video_shot_window_test_data_param().batch_size(), 1, 1, 1);
    this->prefetch_label_.Reshape(this->layer_param_.video_shot_window_test_data_param().batch_size(),
        1, 1, 1);
  }

  // Output datum size
  this->datum_channels_ = context_size_ + positive_size_ + negative_size_;
  this->datum_height_ = feature_size_;
  this->datum_width_ = 1;
  this->datum_size_ = feature_size_ * this->datum_channels_;

}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void VideoShotWindowTestDataLayer<Dtype>::InternalThreadEntry() {

  //LOG(INFO) << "In internal therad .............";
  //LOG(INFO) << "Reading item id ......." << item_id << "/" << batch_size;

  TestVideoShotWindows test_shot_windows;
  CHECK(this->prefetch_data_.count());

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }

  const int batch_size = this->layer_param_.video_shot_window_test_data_param().batch_size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {


    // get a blob
    switch (this->layer_param_.video_shot_window_test_data_param().backend()) {

    case VideoShotWindowTestDataParameter_DB_LEVELDB:
      CHECK(iter_);
      CHECK(iter_->Valid());
      test_shot_windows.ParseFromString(iter_->value().ToString());
      break;
    case VideoShotWindowTestDataParameter_DB_LMDB:
      // Perhaps this is unecessay, we could directly take mdb_value_
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
      test_shot_windows.ParseFromArray(mdb_value_.mv_data,
          mdb_value_.mv_size);
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }

    // Check to ensure that the shot window is valid
    CHECK(test_shot_windows.has_video_id()) << "No video id found for shot window";

    if (this->layer_param_.video_shot_window_test_data_param().include_positives()) {
      CHECK_EQ(test_shot_windows.positive_shot_id_size(), positive_size_);
    }
    CHECK_EQ(test_shot_windows.context_shot_words_size(), context_size_);
    
    if (this->layer_param_.video_shot_window_test_data_param().include_positives()) {
      CHECK_EQ(test_shot_windows.positive_shot_words_size(), positive_size_);
    }
    
    if (this->layer_param_.video_shot_window_test_data_param().include_negatives()) {
      CHECK_EQ(test_shot_windows.negative_shot_words_size(), negative_size_);
    }


 
    // Output the context data to the next channels blob
    // (num_batch * context_size_ * feature_size_ * 1)
    for (int context_id = 0;
        context_id < (context_size_); ++context_id) {
      for (int feature_id = 0; feature_id < this->datum_height_; ++feature_id) {
         top_data[(item_id*this->datum_channels_ + context_id) * this->datum_height_ + feature_id] =
           test_shot_windows.context_shot_words(context_id).float_data(feature_id);
      }
    }

    // The first  positive_size_ channels contains the positive words
    for (int posid = context_size_; posid < (context_size_ + positive_size_); ++posid) {
      for (int feature_id = 0; feature_id < this->datum_height_; ++feature_id) {
        top_data[(item_id*this->datum_channels_ + posid) *this->datum_height_ + feature_id] =
             test_shot_windows.positive_shot_words(posid-context_size_).float_data(feature_id);
      }
    }
    
    // Output the negative data to the next channels blob
    for (int negid = positive_size_ + context_size_;
        negid < (context_size_ + positive_size_ + negative_size_); ++negid) {
      for (int feature_id = 0; feature_id < this->datum_height_; ++feature_id) {
         top_data[(item_id*this->datum_channels_ + negid) * this->datum_height_ + feature_id] =
           test_shot_windows.negative_shot_words(negid-context_size_-positive_size_).float_data(feature_id);
      }
    }


    if (this->output_labels_) {
      top_label[item_id] = test_shot_windows.video_id();
      if (this->layer_param_.video_shot_window_test_data_param().display_all_ids()) {
        LOG(WARNING) << "Item-id:Video-id:Shot-id:" << item_id
          << ":" << test_shot_windows.video_id() << ":"
          << test_shot_windows.positive_shot_id(0);
      }
    }

    // go to the next iter
    switch (this->layer_param_.video_shot_window_test_data_param().backend()) {
    case VideoShotWindowTestDataParameter_DB_LEVELDB:
      iter_->Next();
      if (!iter_->Valid()) {
        // We have reached the end. Restart from the first.
        LOG(INFO) << "Restarting data prefetching from start.";
        iter_->SeekToFirst();
      }
      break;
    case VideoShotWindowTestDataParameter_DB_LMDB:
      if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
        // We have reached the end. Restart from the first.
        //LOG(INFO) << "Restarting data prefetching from start.";
        CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                &mdb_value_, MDB_FIRST), MDB_SUCCESS);
      }
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }
  }
}

INSTANTIATE_CLASS(VideoShotWindowTestDataLayer);

}  // namespace caffe
