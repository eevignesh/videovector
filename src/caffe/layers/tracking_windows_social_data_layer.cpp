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
#include "caffe/proto/tracking_windows.pb.h"

using tracking_windows::TrackPositions;
using tracking_windows::TrackingWindow;

namespace caffe {

template <typename Dtype>
TrackingWindowsSocialDataLayer<Dtype>::~TrackingWindowsSocialDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.tracking_windows_data_param().backend()) {
  case TrackingWindowsDataParameter_DB_LEVELDB:
    break;  // do nothing
  case TrackingWindowsDataParameter_DB_LMDB:
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
void TrackingWindowsSocialDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  if (top->size() >= 6) {
    output_labels_ = true;
  } else {
    output_labels_ = false;
  }

  if (top->size() >= 7) {
    output_scene_ids_ = true;
  } else {
    output_scene_ids_ = false;
  }

  DataLayerSetUp(bottom, top);
}

template <typename Dtype>
void TrackingWindowsSocialDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {


  temporal_observed_size_ = this->layer_param_.tracking_windows_data_param().temporal_observed_size();
  temporal_predicted_size_ = this->layer_param_.tracking_windows_data_param().temporal_predicted_size();
  batch_size_ = this->layer_param_.tracking_windows_data_param().batch_size();
  use_static_scene_ = this->layer_param_.tracking_windows_data_param().use_static_scene();
  num_positions_ = 0;
  max_number_positions_ = this->layer_param_.tracking_windows_data_param().max_number_positions();
  // Initialize DB
  switch (this->layer_param_.tracking_windows_data_param().backend()) {
  
  case TrackingWindowsDataParameter_DB_LEVELDB:
    {
      leveldb::DB* db_temp;
      leveldb::Options options = GetLevelDBOptions();
      options.create_if_missing = false;
      LOG(INFO) << "Opening leveldb " << this->layer_param_.tracking_windows_data_param().source();
      leveldb::Status status = leveldb::DB::Open(
          options, this->layer_param_.tracking_windows_data_param().source(), &db_temp);
      CHECK(status.ok()) << "Failed to open leveldb "
                         << this->layer_param_.tracking_windows_data_param().source() << std::endl
                         << status.ToString();
      db_.reset(db_temp);
      iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
      iter_->SeekToFirst();
    }

    break;
  case TrackingWindowsDataParameter_DB_LMDB:
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.tracking_windows_data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.tracking_windows_data_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";

    break;
  }

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.tracking_windows_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.tracking_windows_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      switch (this->layer_param_.tracking_windows_data_param().backend()) {
      case TrackingWindowsDataParameter_DB_LEVELDB:
        iter_->Next();
        if (!iter_->Valid()) {
          iter_->SeekToFirst();
        }
        break;
      case TrackingWindowsDataParameter_DB_LMDB:
        if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
            != MDB_SUCCESS) {
          CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
                   MDB_FIRST), MDB_SUCCESS);
        }
        break;
      default:
        LOG(FATAL) << "Unknown database backend";
      }
    }
  }


  // Read a data point, and use it to initialize the top blob.
  TrackingWindow track_window;
  switch (this->layer_param_.tracking_windows_data_param().backend()) {
  case TrackingWindowsDataParameter_DB_LEVELDB:
    track_window.ParseFromString(iter_->value().ToString());
    break;
  case TrackingWindowsDataParameter_DB_LMDB:
    track_window.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  CHECK_GE(track_window.observed_time_size(), temporal_observed_size_);
  CHECK_GE(track_window.prediction_time_size(), temporal_predicted_size_);

  int feat_size_orig = 2;
  if (this->layer_param_.tracking_windows_data_param().encoder_bit()) {
    feat_size_orig = 3;
  }

  if (use_static_scene_) {
    feature_size_observed_ = feat_size_orig + track_window.track_positions(0).static_scene().float_data_size();
    feature_size_predicted_ = feat_size_orig + track_window.track_positions(0).static_scene().float_data_size();
  } else {
    feature_size_observed_ = feat_size_orig;
    feature_size_predicted_ = feat_size_orig;
  }

  CHECK_GE(batch_size_, 1);

  // We use a small hack: channels = context_size_+1, height = feature_size_
  (*top)[0]->Reshape(temporal_observed_size_,
        batch_size_, feature_size_observed_, 1);
  (*top)[1]->Reshape(1,
        batch_size_, 2, 1);
  (*top)[2]->Reshape(temporal_predicted_size_,
        batch_size_, feature_size_predicted_, 1);
  (*top)[3]->Reshape(1,
        batch_size_, batch_size_, 1);
  (*top)[4]->Reshape(1,
        batch_size_, 1, 1);

  if (this->output_labels_) {
    (*top)[5]->Reshape(temporal_predicted_size_,
        batch_size_, 2, 1);
    this->prefetch_label_.Reshape(temporal_predicted_size_,
        batch_size_, 2, 1);

  }

  if (this->output_scene_ids_) {
    (*top)[6]->Reshape(1,
        batch_size_, 1, 1);
    this->prefetch_scene_.Reshape(1,
        batch_size_, 1, 1);
  }

  this->prefetch_data_0_.Reshape(temporal_observed_size_,
      batch_size_, feature_size_observed_, 1);
  this->prefetch_data_1_.Reshape(1,
      batch_size_, 2, 1);
  this->prefetch_data_2_.Reshape(temporal_predicted_size_,
      batch_size_, feature_size_predicted_, 1);
  this->prefetch_data_3_.Reshape(1,
      batch_size_, batch_size_, 1);
  this->prefetch_data_4_.Reshape(1,
      batch_size_, 1, 1);

  LOG(INFO) << "output data 0 size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  LOG(INFO) << "output data 1 size: " << (*top)[1]->num() << ","
      << (*top)[1]->channels() << "," << (*top)[1]->height() << ","
      << (*top)[1]->width();
  LOG(INFO) << "output data 2 size: " << (*top)[2]->num() << ","
      << (*top)[2]->channels() << "," << (*top)[2]->height() << ","
      << (*top)[2]->width();
  LOG(INFO) << "output data 3 size: " << (*top)[3]->num() << ","
      << (*top)[3]->channels() << "," << (*top)[3]->height() << ","
      << (*top)[3]->width();
  LOG(INFO) << "output data 4 size: " << (*top)[4]->num() << ","
      << (*top)[4]->channels() << "," << (*top)[4]->height() << ","
      << (*top)[4]->width();

  prev_tracking_window_ = track_window;
  prev_track_id_ = 0;
}


// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void TrackingWindowsSocialDataLayer<Dtype>::InternalThreadEntry() {

  TrackingWindow tracking_window;
  TrackPositions track_position;

  Dtype* top_data_0 = this->prefetch_data_0_.mutable_cpu_data();
  Dtype* mean_data = this->prefetch_data_1_.mutable_cpu_data();
  Dtype* top_data_2 = this->prefetch_data_2_.mutable_cpu_data();
  Dtype* group_data = this->prefetch_data_3_.mutable_cpu_data();
  Dtype* is_valid = this->prefetch_data_4_.mutable_cpu_data();

  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }

  Dtype* top_scene = NULL;  // suppress warnings about uninitialized variables
  if (this->output_scene_ids_) {
    top_scene = this->prefetch_scene_.mutable_cpu_data();
  }


  caffe_set<Dtype>(this->prefetch_data_0_.count(), (Dtype)0., top_data_0);
  caffe_set<Dtype>(this->prefetch_data_1_.count(), (Dtype)0., mean_data);
  caffe_set<Dtype>(this->prefetch_data_2_.count(), (Dtype)0., top_data_2);
  caffe_set<Dtype>(this->prefetch_data_3_.count(), (Dtype)0., group_data);
  caffe_set<Dtype>(this->prefetch_data_4_.count(), (Dtype)0., is_valid);

  int num_positions_set = 0;
  for (int i = 0; i < batch_size_; ++i) {
    while (prev_track_id_ >= prev_tracking_window_.track_positions_size() ||
           prev_tracking_window_.track_positions_size() > batch_size_) {

    // go to the next iter
    switch (this->layer_param_.tracking_windows_data_param().backend()) {
      case TrackingWindowsDataParameter_DB_LEVELDB:
        iter_->Next();
        if (!iter_->Valid() || (num_positions_==0 && max_number_positions_ > 0 && i==0)) {
          // We have reached the end. Restart from the first.
          LOG(INFO) << "Restarting data prefetching from start."
            << "prev_id: " << prev_track_id_
            << "prev_size: " << prev_tracking_window_.track_positions_size()
            << "num_positions: " << num_positions_
            << "max_number_positions: " << max_number_positions_;
          iter_->SeekToFirst();
        }
        break;
      case TrackingWindowsDataParameter_DB_LMDB:
        if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
                &mdb_value_, MDB_NEXT) != MDB_SUCCESS
            || (num_positions_==0 && max_number_positions_ > 0 && i==0)
            ) {
          // We have reached the end. Restart from the first.
          LOG(INFO) << "Restarting data prefetching from start."
            << "prev_id: " << prev_track_id_
            << "prev_size: " << prev_tracking_window_.track_positions_size()
            << "num_positions: " << num_positions_
            << "max_number_positions: " << max_number_positions_;

          CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                  &mdb_value_, MDB_FIRST), MDB_SUCCESS);
        }
        break;
      default:
        LOG(FATAL) << "Unknown database backend";
      }

       // get a blob
      switch (this->layer_param_.tracking_windows_data_param().backend()) {
        case TrackingWindowsDataParameter_DB_LEVELDB:
          CHECK(iter_);
          CHECK(iter_->Valid());
          tracking_window.ParseFromString(iter_->value().ToString());
          break;
        case TrackingWindowsDataParameter_DB_LMDB:
          // Perhaps this is unecessay, we could directly take mdb_value_
          CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                  &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
          tracking_window.ParseFromArray(mdb_value_.mv_data,
              mdb_value_.mv_size);
          break;
        default:
          LOG(FATAL) << "Unknown database backend";
      }
      prev_tracking_window_ = tracking_window;
      prev_track_id_ = 0;
    }

    //CHECK_GE(prev_tracking_window_.track_positions_size(), prev_track_id_+1);

    if (prev_track_id_ == 0) {
      if ((batch_size_-i) < prev_tracking_window_.track_positions_size()) {
        break;
      } else {
        // Identify which of the tracks fall into the same group
        for (int j = 0; j < prev_tracking_window_.track_positions_size(); ++j) {
          caffe_set<Dtype>(prev_tracking_window_.track_positions_size(), (Dtype)1.,
            group_data + ((i+j)*batch_size_ + i));
          caffe_set<Dtype>(1, (Dtype)0., group_data + ((i+j)*batch_size_ + (i+j)));
        }
      }
    }

    //LOG(ERROR) << prev_tracking_window_.track_positions_size() << ":"
    //  << prev_track_id_;
    track_position = prev_tracking_window_.track_positions(prev_track_id_);

    CHECK_GE(track_position.x_size(), temporal_observed_size_ + temporal_predicted_size_);
    CHECK_GE(track_position.y_size(), temporal_observed_size_ + temporal_predicted_size_);

    Dtype mean_x = (Dtype)track_position.x(temporal_observed_size_-1)*this->layer_param_.tracking_windows_data_param().track_scale();
    Dtype mean_y = (Dtype)track_position.y(temporal_observed_size_-1)*this->layer_param_.tracking_windows_data_param().track_scale();

    *(mean_data + this->prefetch_data_1_.offset(0, i, 0, 0)) = mean_x;
    *(mean_data + this->prefetch_data_1_.offset(0, i, 1, 0)) = mean_y;
    *(is_valid + this->prefetch_data_4_.offset(0, i, 0, 0)) = (Dtype)1.0;

    for (int t = 0; t < temporal_observed_size_; ++t) {
      *(top_data_0 + this->prefetch_data_0_.offset(t, i, 0, 0)) = (Dtype)track_position.x(t)*this->layer_param_.tracking_windows_data_param().track_scale()- mean_x;
      *(top_data_0 + this->prefetch_data_0_.offset(t, i, 1, 0)) = (Dtype)track_position.y(t)*this->layer_param_.tracking_windows_data_param().track_scale()- mean_y;
      
      if (this->layer_param_.tracking_windows_data_param().encoder_bit()) {
        *(top_data_0 + this->prefetch_data_0_.offset(t, i, 2, 0)) = (Dtype)0.;
      }

      int offset_bit = 2;
      if (use_static_scene_) {
        if (this->layer_param_.tracking_windows_data_param().encoder_bit()) {
          offset_bit = 3;
        }

        if (t==0) {
          for (int f = 0; f < feature_size_observed_-offset_bit; ++f) {
            *(top_data_0 + this->prefetch_data_0_.offset(t, i, offset_bit+f, 0)) =
              track_position.static_scene().float_data(f);
          }
        } else {
          caffe_copy<Dtype>(feature_size_observed_-offset_bit,
              (top_data_0 + this->prefetch_data_0_.offset(0, i, offset_bit, 0)) ,
              top_data_0 + this->prefetch_data_0_.offset(t, i, offset_bit, 0));
        }
      }
    }

    /*if (!use_static_scene_) {
      caffe_set<Dtype>(batch_size_*temporal_predicted_size_*feature_size_predicted_,
          (Dtype)0.0, top_data_2);
    }*/


    for (int t = 0; t < temporal_predicted_size_; ++t) {

      if (output_labels_) {
        *(top_label + this->prefetch_label_.offset(t, i, 0, 0)) = (Dtype)track_position.x(t + temporal_observed_size_)*this->layer_param_.tracking_windows_data_param().track_scale() - mean_x;
        *(top_label + this->prefetch_label_.offset(t, i, 1, 0)) = (Dtype)track_position.y(t + temporal_observed_size_)*this->layer_param_.tracking_windows_data_param().track_scale() - mean_y;
      }

      if (output_scene_ids_ && t==0) {
        *(top_scene + this->prefetch_scene_.offset(0, i, 0, 0)) =
          (Dtype)track_position.id();
          //(Dtype)prev_tracking_window_.scene_id();
      }

      if (use_static_scene_) {

        if (this->layer_param_.tracking_windows_data_param().encoder_bit()) {
          caffe_copy<Dtype>(feature_size_predicted_-3,
              (top_data_0 + this->prefetch_data_0_.offset(0, i, 3, 0)),
              top_data_2 + this->prefetch_data_2_.offset(t, i, 3, 0));
        } else {
          caffe_copy<Dtype>(feature_size_predicted_-2,
              (top_data_0 + this->prefetch_data_0_.offset(0, i, 2, 0)),
              top_data_2 + this->prefetch_data_2_.offset(t, i, 2, 0));
        }
      } else {
        // TODO: change this later
        *(top_data_2 + this->prefetch_data_2_.offset(t, i, 0, 0)) = (Dtype)0.; //(Dtype)track_position.x(temporal_observed_size_-1);
        *(top_data_2 + this->prefetch_data_2_.offset(t, i, 1, 0)) = (Dtype)0.; //(Dtype)track_position.y(temporal_observed_size_-1);

        if (this->layer_param_.tracking_windows_data_param().encoder_bit()) {
          *(top_data_2 + this->prefetch_data_2_.offset(t, i, 2, 0)) = (Dtype)1.;
        }
      }
    }
    
    prev_track_id_++;
    num_positions_set++;
  }

  /*for (int i = 0; i < batch_size_; ++i) {
    LOG(INFO) << "num_pos_set: " << num_positions_set
      << " : " << is_valid[i];
  }*/

  // Once a certain set of iterations have been reached
  // roll back and start the iterations again
  if (max_number_positions_ > 0) {
    num_positions_++;
    if (num_positions_ >= max_number_positions_) {
      num_positions_ = 0;
      prev_track_id_ = prev_tracking_window_.track_positions_size() + 30;
    }
  }

}

template <typename Dtype>
void TrackingWindowsSocialDataLayer<Dtype>::CreatePrefetchThread() {
  this->phase_ = Caffe::phase();
  CHECK(StartInternalThread()) << "Thread execution failed";
}

template <typename Dtype>
void TrackingWindowsSocialDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}

template <typename Dtype>
void TrackingWindowsSocialDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_0_.count(), prefetch_data_0_.cpu_data(),
             (*top)[0]->mutable_cpu_data());
  caffe_copy(prefetch_data_1_.count(), prefetch_data_1_.cpu_data(),
             (*top)[1]->mutable_cpu_data());
  caffe_copy(prefetch_data_2_.count(), prefetch_data_2_.cpu_data(),
             (*top)[2]->mutable_cpu_data());
  caffe_copy(prefetch_data_3_.count(), prefetch_data_3_.cpu_data(),
             (*top)[3]->mutable_cpu_data());
  caffe_copy(prefetch_data_4_.count(), prefetch_data_4_.cpu_data(),
             (*top)[4]->mutable_cpu_data());

  if (this->output_labels_) {
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
               (*top)[5]->mutable_cpu_data());
  }

  if (this->output_scene_ids_) {
    caffe_copy(prefetch_scene_.count(), prefetch_scene_.cpu_data(),
               (*top)[6]->mutable_cpu_data());
  }

  // Start a new prefetch thread
  CreatePrefetchThread();
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(TrackingWindowsSocialDataLayer, Forward);
#endif

INSTANTIATE_CLASS(TrackingWindowsSocialDataLayer);

}  // namespace caffe
