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

using video_shot_sentences::VideoShotWindow;

// TODO: Make this a parameter
DEFINE_int32(max_negative_tries, 100, "Maximum number of tries for adding negatives");

namespace caffe {

template <typename Dtype>
inline int VideoShotWindowDataLayer<Dtype>::AddToBuffer(const Datum datum) {
  Dtype* negative_mutable_data = negatives_.mutable_cpu_data();
  if ((rand() % 100) < negative_swap_percentage_) {
    // Randomly remove a negative and swap it out with this
    int rand_id = rand() % max_buffer_size_;
    
    for (int i = 0; i < feature_size_; ++i) {
      negative_mutable_data[(rand_id * feature_size_) + i] = datum.float_data(i);
    }
    return rand_id;
  }
  return -1;
}

// This algorithm shuffles the negative list so that the top n
// elements are randomly chosen. (Fisher-Yates Algo.)
template <typename Dtype>
inline void VideoShotWindowDataLayer<Dtype>::RandomShuffleTopids(int n) {
  random_unique(buffer_ids_.begin(), buffer_ids_.end(), n);
}

template <typename Dtype>
VideoShotWindowDataLayer<Dtype>::~VideoShotWindowDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.video_shot_window_data_param().backend()) {
  case VideoShotWindowDataParameter_DB_LEVELDB:
    break;  // do nothing
  case VideoShotWindowDataParameter_DB_LMDB:
    mdb_cursor_close(mdb_cursor_);
    mdb_close(mdb_env_, mdb_dbi_);
    mdb_txn_abort(mdb_txn_);
    mdb_env_close(mdb_env_);
    break;
  case VideoShotWindowDataParameter_DB_VIDEO_ID_TEXT:
    next_video_id_ = 0;
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

template <typename Dtype>
void VideoShotWindowDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  // Set up the negative buffer
  num_negative_samples_ = this->layer_param_.video_shot_window_data_param().num_negative_samples();
  max_buffer_size_ = 0;

  if (num_negative_samples_ > 0) {
    max_buffer_size_ = this->layer_param_.video_shot_window_data_param().max_buffer_size();
    negative_swap_percentage_ = this->layer_param_.video_shot_window_data_param().negative_swap_percentage();
    CHECK_GE(negative_swap_percentage_, 0) << "Swap percentage should be greater than 0";
    CHECK_LE(negative_swap_percentage_, 99) << "Swap percentage should be less than 100";
    for (int i = 0; i < max_buffer_size_; ++i) {
      buffer_ids_.push_back(i);
    }
  }

  // Initialize DB
  switch (this->layer_param_.video_shot_window_data_param().backend()) {
  case VideoShotWindowDataParameter_DB_LEVELDB:
    {
      leveldb::DB* db_temp;
      leveldb::Options options = GetLevelDBOptions();
      options.create_if_missing = false;
      LOG(INFO) << "Opening leveldb " << this->layer_param_.video_shot_window_data_param().source();
      leveldb::Status status = leveldb::DB::Open(
          options, this->layer_param_.video_shot_window_data_param().source(), &db_temp);
      CHECK(status.ok()) << "Failed to open leveldb "
                         << this->layer_param_.video_shot_window_data_param().source() << std::endl
                         << status.ToString();
      db_.reset(db_temp);
      iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
      iter_->SeekToFirst();
    }

    // Negative data buffer if needed
    if (this->layer_param_.video_shot_window_data_param().negative_dataset() != "") {
      leveldb::DB* db_temp;
      leveldb::Options options = GetLevelDBOptions();
      options.create_if_missing = false;
      LOG(INFO) << "Opening leveldb " << this->layer_param_.video_shot_window_data_param().negative_dataset();
      leveldb::Status status = leveldb::DB::Open(
          options, this->layer_param_.video_shot_window_data_param().negative_dataset(), &db_temp);
      CHECK(status.ok()) << "Failed to open leveldb "
                         << this->layer_param_.video_shot_window_data_param().negative_dataset()
                         << std::endl
                         << status.ToString();
      db_neg_.reset(db_temp);
      iter_neg_.reset(db_neg_->NewIterator(leveldb::ReadOptions()));
      iter_neg_->SeekToFirst();
    }
    break;
  case VideoShotWindowDataParameter_DB_LMDB:
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.video_shot_window_data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.video_shot_window_data_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";

    // Negative buffer if provided
    if (this->layer_param_.video_shot_window_data_param().negative_dataset() != "") {
      CHECK_EQ(mdb_env_create(&mdb_env_neg_), MDB_SUCCESS) << "mdb_env_create failed";
      CHECK_EQ(mdb_env_set_mapsize(mdb_env_neg_, 1099511627776), MDB_SUCCESS);  // 1TB
      CHECK_EQ(mdb_env_open(mdb_env_neg_,
               this->layer_param_.video_shot_window_data_param().negative_dataset().c_str(),
               MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
      CHECK_EQ(mdb_txn_begin(mdb_env_neg_, NULL, MDB_RDONLY, &mdb_txn_neg_), MDB_SUCCESS)
          << "mdb_txn_begin failed";
      CHECK_EQ(mdb_open(mdb_txn_neg_, NULL, 0, &mdb_dbi_neg_), MDB_SUCCESS)
          << "mdb_open failed";
      CHECK_EQ(mdb_cursor_open(mdb_txn_neg_, mdb_dbi_neg_, &mdb_cursor_neg_), MDB_SUCCESS)
          << "mdb_cursor_open failed";
      LOG(INFO) << "Opening lmdb " << this->layer_param_.video_shot_window_data_param().negative_dataset();
      CHECK_EQ(mdb_cursor_get(mdb_cursor_neg_, &mdb_key_neg_, &mdb_value_neg_, MDB_FIRST),
          MDB_SUCCESS) << "mdb_cursor_get failed";
    }
    break;
  case VideoShotWindowDataParameter_DB_VIDEO_ID_TEXT:
    CHECK_EQ(num_negative_samples_,0);
    const string& source_text = this->layer_param_.video_shot_window_data_param().source();
    LOG(INFO) << "Opening text file " << source_text;
    std::ifstream infile(source_text.c_str());
    int video_id;
    while (infile >> video_id) {
      test_video_ids_.push_back(video_id);
    }
    next_video_id_ = 0;
    break;
  //default:
  //  LOG(FATAL) << "Unknown database backend";
  }

  // If only reading video ids, then exit
  if (this->layer_param_.video_shot_window_data_param().backend() == VideoShotWindowDataParameter_DB_VIDEO_ID_TEXT) {
    CHECK_EQ(top->size(), 1);
    (*top)[0]->Reshape(
      this->layer_param_.video_shot_window_data_param().batch_size(), 1,
        1, 1);
    this->prefetch_data_.Reshape(this->layer_param_.video_shot_window_data_param().batch_size(),
      1, 1, 1);
    this->datum_channels_ = 1;
    this->datum_height_ = 1;
    this->datum_width_ = 1;
    this->datum_size_ = 1;
    return;
  }

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.video_shot_window_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.video_shot_window_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      switch (this->layer_param_.video_shot_window_data_param().backend()) {
      case VideoShotWindowDataParameter_DB_LEVELDB:
        iter_->Next();
        if (!iter_->Valid()) {
          iter_->SeekToFirst();
        }
        break;
      case VideoShotWindowDataParameter_DB_LMDB:
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
  VideoShotWindow shot_window;
  switch (this->layer_param_.video_shot_window_data_param().backend()) {
  case VideoShotWindowDataParameter_DB_LEVELDB:
    shot_window.ParseFromString(iter_->value().ToString());
    break;
  case VideoShotWindowDataParameter_DB_LMDB:
    shot_window.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  //LOG(INFO) << shot_window.DebugString();

  // We expect one-d fetures only
  feature_size_ = shot_window.target_shot_word().float_data_size();
  context_size_ = shot_window.context_shot_words_size();
  //CHECK_LE(shot_window.target_shot_word().height(), 1);
  //CHECK_LE(shot_window.target_shot_word().width(), 1);
  CHECK_GE(feature_size_, 1);
  CHECK_GE(context_size_, 1); // need atleast a context of two words

  // We use a small hack: channels = context_size_+1, height = feature_size_
  (*top)[0]->Reshape(
      this->layer_param_.video_shot_window_data_param().batch_size(),
        context_size_+1+num_negative_samples_, // context-features + target featuers + negatives
        feature_size_, 1);
  this->prefetch_data_.Reshape(this->layer_param_.video_shot_window_data_param().batch_size(),
      context_size_+1+num_negative_samples_, feature_size_, 1);

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();

  // Video id if needed
  if (this->output_labels_) {
    (*top)[1]->Reshape(this->layer_param_.video_shot_window_data_param().batch_size(), 1, 1, 1);
    this->prefetch_label_.Reshape(this->layer_param_.video_shot_window_data_param().batch_size(),
        1, 1, 1);
  }

  // Allocate the negative buffer
  Dtype* negatives_mutable_data = NULL;
  if (num_negative_samples_ > 0) {
    negatives_.Reshape(max_buffer_size_, 1, feature_size_, 1);
    negatives_mutable_data = negatives_.mutable_cpu_data();
  }

  // Initializing the negative samples with first num_negative_samples_
  // target shot words in the dataset
   LOG(INFO) << "Initializing the negative buffer";
   int num_negatives_added = 0;

   for (int nid = 0; nid < (FLAGS_max_negative_tries*max_buffer_size_) ; ++nid) {
    if (num_negatives_added % 1000 == 0) {
      LOG(INFO) << "Added " << num_negatives_added << " negatives to buffer";
    }

    switch (this->layer_param_.video_shot_window_data_param().backend()) {
    case VideoShotWindowDataParameter_DB_LEVELDB:

      if (this->layer_param_.video_shot_window_data_param().negative_dataset() != "") {
        CHECK(iter_neg_);
        CHECK(iter_neg_->Valid());
        shot_window.ParseFromString(iter_neg_->value().ToString());
        iter_neg_->Next();
        if (!iter_neg_->Valid()) {
          iter_neg_->SeekToFirst();
        }
      } else {
        CHECK(iter_);
        CHECK(iter_->Valid());
        shot_window.ParseFromString(iter_->value().ToString());
        iter_->Next();
        if (!iter_->Valid()) {
          iter_->SeekToFirst();
        }
      }
      break;

    case VideoShotWindowDataParameter_DB_LMDB:
      if (this->layer_param_.video_shot_window_data_param().negative_dataset() != "") {

        CHECK_EQ(mdb_cursor_get(mdb_cursor_neg_, &mdb_key_neg_,
              &mdb_value_neg_, MDB_GET_CURRENT), MDB_SUCCESS);
        shot_window.ParseFromArray(mdb_value_neg_.mv_data,
          mdb_value_neg_.mv_size);
        if (mdb_cursor_get(mdb_cursor_neg_, &mdb_key_neg_, &mdb_value_neg_, MDB_NEXT)
            != MDB_SUCCESS) {
          CHECK_EQ(mdb_cursor_get(mdb_cursor_neg_, &mdb_key_neg_, &mdb_value_neg_,
                   MDB_FIRST), MDB_SUCCESS);
        }
      } else {
        CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
        shot_window.ParseFromArray(mdb_value_.mv_data,
          mdb_value_.mv_size);
        if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
            != MDB_SUCCESS) {
          CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
                   MDB_FIRST), MDB_SUCCESS);
        }
      }
      break;

    default:
      LOG(FATAL) << "Unknown database backend";
    }

    string negative_key = stringprintf("%d:%d", shot_window.video_id(), shot_window.shot_id());
    if (negative_keys_set_.find(negative_key) == negative_keys_set_.end()) {
      // Copy data into negative samples
      for (int f = 0; f < feature_size_; ++f) {
        negatives_mutable_data[num_negatives_added*feature_size_ + f] =
          shot_window.target_shot_word().float_data(f);
      }
      negative_id_to_key_.push_back(negative_key);
      negative_keys_set_.insert(negative_key);
      num_negatives_added++;
    }

    if (num_negatives_added >= max_buffer_size_) {
      LOG(INFO) << "Successfully added " << max_buffer_size_ << " negatives.";
      break;
    }
  }

  CHECK_EQ(num_negatives_added, max_buffer_size_) << "Could not add requested number of negatives";

  // Output datum size
  this->datum_channels_ = context_size_ + 1 + num_negative_samples_;
  this->datum_height_ = feature_size_;
  this->datum_width_ = 1;
  this->datum_size_ = feature_size_ * this->datum_channels_;

  // Close the negative dataset buffer
  if (this->layer_param_.video_shot_window_data_param().negative_dataset() != "") {
    switch (this->layer_param_.data_param().backend()) {
      case DataParameter_DB_LEVELDB:
        break;  // do nothing
      case DataParameter_DB_LMDB:
        mdb_cursor_close(mdb_cursor_neg_);
        mdb_close(mdb_env_neg_, mdb_dbi_neg_);
        mdb_txn_abort(mdb_txn_neg_);
        mdb_env_close(mdb_env_neg_);
        break;
      default:
        LOG(FATAL) << "Unknown database backend";
    }
  }
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void VideoShotWindowDataLayer<Dtype>::InternalThreadEntry() {
  VideoShotWindow shot_window;
  CHECK(this->prefetch_data_.count());

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.video_shot_window_data_param().batch_size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {

    if (this->layer_param_.video_shot_window_data_param().backend() == VideoShotWindowDataParameter_DB_VIDEO_ID_TEXT) {
      top_data[item_id] = test_video_ids_[next_video_id_];
      next_video_id_ = (next_video_id_ + 1) % test_video_ids_.size();
      continue;
    }

    // get a blob
    switch (this->layer_param_.video_shot_window_data_param().backend()) {
    case VideoShotWindowDataParameter_DB_LEVELDB:
      CHECK(iter_);
      CHECK(iter_->Valid());
      shot_window.ParseFromString(iter_->value().ToString());
      break;
    case VideoShotWindowDataParameter_DB_LMDB:
      // Perhaps this is unecessay, we could directly take mdb_value_
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
      shot_window.ParseFromArray(mdb_value_.mv_data,
          mdb_value_.mv_size);
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }

    // Check to ensure that the shot window is valid
    CHECK(shot_window.has_video_id()) << "No video id found for shot window";
    CHECK(shot_window.has_target_shot_word()) << "No target shot word: "
                                              << shot_window.video_id();
    CHECK_EQ(shot_window.context_shot_words_size(), (this->datum_channels_-1 -num_negative_samples_))
      << "Insufficient context words for: " << shot_window.video_id();

    CHECK_EQ(shot_window.target_shot_word().float_data_size(), this->datum_height_)
      << "Wrong feature size: " << shot_window.video_id();


    // The first channel contains the actual target word
    for (int feature_id = 0; feature_id < this->datum_height_; ++feature_id) {
      top_data[item_id*this->datum_channels_*this->datum_height_ + feature_id] =
           shot_window.target_shot_word().float_data(feature_id);
    }
  
    // Output the context data to the next channels blob
    // (num_batch * context_size_ * feature_size_ * 1)
    for (int context_id = 1; context_id < (this->context_size_ + 1); ++context_id) {
      CHECK_EQ(shot_window.target_shot_word().float_data_size(), this->datum_height_)
        << "Wrong context feature size: " << shot_window.video_id() << ":" << (context_id-1);
      for (int feature_id = 0; feature_id < this->datum_height_; ++feature_id) {
         top_data[(item_id*this->datum_channels_ + context_id) * this->datum_height_ + feature_id] =
           shot_window.context_shot_words(context_id-1).float_data(feature_id);
      }
    }


    // Add the word to negative buffer
    if (num_negative_samples_ > 0) {

      // Swap values in buffer if needed
      if (negative_swap_percentage_ > 0) {
        string new_negative_key = stringprintf("%d:%d", shot_window.video_id(),
            shot_window.shot_id());
        // Replace only if the shot is already not in the negative buffer
        if (negative_keys_set_.find(new_negative_key) == negative_keys_set_.end()) {
          int negative_pos = AddToBuffer(shot_window.target_shot_word());
          if (negative_pos >= 0) {
            string old_negative_key = negative_id_to_key_[negative_pos];
            negative_id_to_key_[negative_pos] = new_negative_key;
            negative_keys_set_.erase(old_negative_key);
            negative_keys_set_.insert(new_negative_key);
          }
        }
      }

      const Dtype* negatives_data = negatives_.cpu_data();
      // Sample the remaining from negatives
      RandomShuffleTopids(num_negative_samples_);
      //TODO: 
      for (int negative_id = (this->context_size_ + 1);
          negative_id < (1 + this->context_size_ + num_negative_samples_); ++negative_id) {
        int neg_id = static_cast<int>(this->buffer_ids_[negative_id - context_size_ + 1]);
        for (int feature_id = 0; feature_id < feature_size_; ++feature_id) {

          top_data[(item_id*this->datum_channels_ + negative_id) * this->datum_height_ + feature_id] =
           negatives_data[neg_id*feature_size_ + feature_id];
        }
        //LOG(INFO) << "neg-id: " << neg_id;
      }
    }

    if (this->output_labels_) {
      top_label[item_id] = shot_window.video_id();
      if (this->layer_param_.video_shot_window_data_param().display_all_ids()) {
        LOG(WARNING) << "Item-id:Video-id:Shot-id:" << item_id << ":" << shot_window.video_id() << ":" << shot_window.shot_id();
      }
    }

    // go to the next iter
    switch (this->layer_param_.video_shot_window_data_param().backend()) {
    case VideoShotWindowDataParameter_DB_LEVELDB:
      iter_->Next();
      if (!iter_->Valid()) {
        // We have reached the end. Restart from the first.
        LOG(INFO) << "Restarting data prefetching from start.";
        iter_->SeekToFirst();
      }
      break;
    case VideoShotWindowDataParameter_DB_LMDB:
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

INSTANTIATE_CLASS(VideoShotWindowDataLayer);

}  // namespace caffe
