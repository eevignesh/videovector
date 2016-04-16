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

namespace caffe {

template <typename Dtype>
FlexibleDataLayer<Dtype>::~FlexibleDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    break;  // do nothing
  case DataParameter_DB_LMDB:
    mdb_cursor_close(mdb_cursor_);
    mdb_close(mdb_env_, mdb_dbi_);
    mdb_txn_abort(mdb_txn_);
    mdb_env_close(mdb_env_);

    mdb_cursor_close(flexi_mdb_cursor_);
    mdb_close(flexi_mdb_env_, flexi_mdb_dbi_);
    mdb_txn_abort(flexi_mdb_txn_);
    mdb_env_close(flexi_mdb_env_);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

template <typename Dtype>
void FlexibleDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Initialize DB
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    {
    leveldb::DB* db_temp;
    leveldb::Options options = GetLevelDBOptions();
    options.create_if_missing = false;
    LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
    leveldb::Status status = leveldb::DB::Open(
        options, this->layer_param_.data_param().source(), &db_temp);
    CHECK(status.ok()) << "Failed to open leveldb "
                       << this->layer_param_.data_param().source() << std::endl
                       << status.ToString();
    db_.reset(db_temp);
    iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
    iter_->SeekToFirst();
    }
    break;
  case DataParameter_DB_LMDB:
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.data_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";

    // flexi-lmdb
    CHECK_EQ(mdb_env_create(&flexi_mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(flexi_mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(flexi_mdb_env_,
             this->layer_param_.flexible_data_param().flexible_source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(flexi_mdb_env_, NULL, MDB_RDONLY, &flexi_mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(flexi_mdb_txn_, NULL, 0, &flexi_mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(flexi_mdb_txn_, flexi_mdb_dbi_, &flexi_mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.flexible_data_param().flexible_source();
    CHECK_EQ(mdb_cursor_get(flexi_mdb_cursor_, &flexi_mdb_key_, &flexi_mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";    
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      switch (this->layer_param_.data_param().backend()) {
      case DataParameter_DB_LEVELDB:
        iter_->Next();
        if (!iter_->Valid()) {
          iter_->SeekToFirst();
        }
        break;
      case DataParameter_DB_LMDB:
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
  Datum datum;
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    datum.ParseFromString(iter_->value().ToString());
    break;
  case DataParameter_DB_LMDB:
    datum.ParseFromArray(flexi_mdb_value_.mv_data, flexi_mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // image
  int flexible_data_length = this->layer_param_.flexible_data_param().forward_len() + this->layer_param_.flexible_data_param().backward_len();
  if (this->layer_param_.flexible_data_param().use_center_datum()) {
    flexible_data_length = flexible_data_length + 1;
  }
  item_channels_ = flexible_data_length * datum.channels();

  int crop_size = this->layer_param_.transform_param().crop_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(this->layer_param_.data_param().batch_size(),
                       item_channels_, crop_size, crop_size);
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        item_channels_, crop_size, crop_size);
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.data_param().batch_size(), item_channels_,
        datum.height(), datum.width());
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        item_channels_, datum.height(), datum.width());
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  if (this->output_labels_) {
    (*top)[1]->Reshape(this->layer_param_.data_param().batch_size(), 1, 1, 1);
    this->prefetch_label_.Reshape(this->layer_param_.data_param().batch_size(),
        1, 1, 1);
  }
  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void FlexibleDataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.data_param().batch_size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {

    // set random cropping and mirroring
    int h_off = 0;
    int w_off = 0;
    bool do_mirror = false;

    const int crop_size = this->layer_param_.transform_param().crop_size();
    if (crop_size) {
      if (this->phase_ == Caffe::TRAIN) {
        h_off = this->data_transformer_.Rand() % (this->datum_height_ - crop_size);
        w_off = this->data_transformer_.Rand() % (this->datum_width_ - crop_size);
      } else {
        h_off = (this->datum_height_ - crop_size) / 2;
        w_off = (this->datum_width_ - crop_size) / 2;
      }
    }
    if (this->layer_param_.transform_param().mirror() && this->data_transformer_.Rand() % 2) {
      do_mirror = true;
    }

    CHECK_GE(h_off,0);
    CHECK_GE(w_off,0);
    CHECK_LE(h_off+crop_size,this->datum_height_);
    CHECK_LE(w_off+crop_size,this->datum_width_);

    std::string item_val;

    // get a blob
    CHECK_EQ(this->layer_param_.data_param().backend(), DataParameter_DB_LMDB);
    switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      CHECK(iter_);
      CHECK(iter_->Valid());
      datum.ParseFromString(iter_->value().ToString());
      break;
    case DataParameter_DB_LMDB:
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
      item_val = std::string(reinterpret_cast<char*>(mdb_value_.mv_data), mdb_value_.mv_size);
      // datum.ParseFromArray(mdb_value_.mv_data,
      //     mdb_value_.mv_size);
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }

    std::string item_key = std::string(reinterpret_cast<char*>(mdb_key_.mv_data), mdb_key_.mv_size);
    std::string flexible_key_base = item_key.substr(9, item_val.size()-9);
    std::string flexible_key_suffix = this->layer_param_.flexible_data_param().flexible_key_suffix();
    int label = atoi(item_val.substr(0,4).c_str());

    std::string flexible_key = flexible_key_base + flexible_key_suffix;
    flexi_mdb_key_.mv_data = reinterpret_cast<void*>(&flexible_key[0]);
    flexi_mdb_key_.mv_size = flexible_key.size();
    CHECK_EQ(mdb_cursor_get(flexi_mdb_cursor_, &flexi_mdb_key_,
              &flexi_mdb_value_, MDB_SET), MDB_SUCCESS);


    const bool use_center_datum = this->layer_param_.flexible_data_param().use_center_datum();
    const int forward_len = this->layer_param_.flexible_data_param().forward_len();
    const int backward_len = this->layer_param_.flexible_data_param().backward_len();
    const int datum_channels = this->datum_channels_;
    const int item_channels = (backward_len + forward_len + (use_center_datum ? 1 : 0)) * datum_channels;
    int channel_start_idx = 0;

    // rewind cursor by backward_len
    for (int flexi_iter = 0; flexi_iter < backward_len; ++flexi_iter) {
      CHECK_EQ(mdb_cursor_get(flexi_mdb_cursor_, &flexi_mdb_key_,
                &flexi_mdb_value_, MDB_PREV), MDB_SUCCESS);
    }

    // read backward_len datums
    for (int flexi_iter = 0; flexi_iter < backward_len; ++flexi_iter) {
      CHECK_EQ(mdb_cursor_get(flexi_mdb_cursor_, &flexi_mdb_key_,
                &flexi_mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
      datum.ParseFromArray(flexi_mdb_value_.mv_data,
          flexi_mdb_value_.mv_size);
      this->data_transformer_.Transform(item_id, datum, this->mean_, top_data, channel_start_idx,
        item_channels, true, do_mirror, h_off, w_off);
      channel_start_idx += datum_channels;
      CHECK_EQ(mdb_cursor_get(flexi_mdb_cursor_, &flexi_mdb_key_,
          &flexi_mdb_value_, MDB_NEXT), MDB_SUCCESS);  
    }    

    // read center datum
    if (use_center_datum) {
      CHECK_EQ(mdb_cursor_get(flexi_mdb_cursor_, &flexi_mdb_key_,
                &flexi_mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
      datum.ParseFromArray(flexi_mdb_value_.mv_data,
          flexi_mdb_value_.mv_size);
      this->data_transformer_.Transform(item_id, datum, this->mean_, top_data, channel_start_idx,
        item_channels, true, do_mirror, h_off, w_off);
      channel_start_idx += datum_channels;
    }
    // forward past center datum regardless if it was read
    // add check in case we're at the end (which is fine if forward_len==0)
    if (forward_len > 0) {
       CHECK_EQ(mdb_cursor_get(flexi_mdb_cursor_, &flexi_mdb_key_,
          &flexi_mdb_value_, MDB_NEXT), MDB_SUCCESS);    
    }
 
    // read forward_len datums
    for (int flexi_iter = 0; flexi_iter < forward_len; ++flexi_iter) {
      CHECK_EQ(mdb_cursor_get(flexi_mdb_cursor_, &flexi_mdb_key_,
                &flexi_mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
      datum.ParseFromArray(flexi_mdb_value_.mv_data,
          flexi_mdb_value_.mv_size);
      this->data_transformer_.Transform(item_id, datum, this->mean_, top_data, channel_start_idx,
        item_channels, true, do_mirror, h_off, w_off);
      channel_start_idx += datum_channels;
      CHECK_EQ(mdb_cursor_get(flexi_mdb_cursor_, &flexi_mdb_key_,
          &flexi_mdb_value_, MDB_NEXT), MDB_SUCCESS);  
    }    

    // Apply data transformations (mirror, scale, crop...)
    //this->data_transformer_.Transform(item_id, datum, this->mean_, top_data);

    if (this->output_labels_) {
      //top_label[item_id] = datum.label();
      top_label[item_id] = label;
    }

    // go to the next iter
    switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      iter_->Next();
      if (!iter_->Valid()) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        iter_->SeekToFirst();
      }
      break;
    case DataParameter_DB_LMDB:
      if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                &mdb_value_, MDB_FIRST), MDB_SUCCESS);
      }
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }
  }
}

INSTANTIATE_CLASS(FlexibleDataLayer);

}  // namespace caffe
