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

using video_shot_sentences::VideoShots;

// TODO: Make this a parameter
DEFINE_int32(max_tries, 100, "Maximum number of tries for adding negatives");

namespace caffe {

template <typename Dtype>
inline int VideoShotsDataLayer<Dtype>::AddToBuffer(const Datum datum) {
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
inline void VideoShotsDataLayer<Dtype>::RandomShuffleTopids(int n) {
  random_unique(buffer_ids_.begin(), buffer_ids_.end(), n);
}

template <typename Dtype>
VideoShotsDataLayer<Dtype>::~VideoShotsDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.video_shots_data_param().backend()) {
  case VideoShotsDataParameter_DB_LEVELDB:
    break;  // do nothing
  case VideoShotsDataParameter_DB_LMDB:
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
void VideoShotsDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  // Set up the negative buffer
  num_negative_samples_ = this->layer_param_.video_shots_data_param().num_negative_samples();

  output_shot_distance_ = this->layer_param_.video_shots_data_param().output_shot_distance();

  max_buffer_size_ = 0;
  max_same_video_negs_ = this->layer_param_.video_shots_data_param().max_same_video_negs();

  if (num_negative_samples_ > 0) {
    max_buffer_size_ = this->layer_param_.video_shots_data_param().max_buffer_size();
    negative_swap_percentage_ = this->layer_param_.video_shots_data_param().negative_swap_percentage();
    CHECK_GE(negative_swap_percentage_, 0) << "Swap percentage should be greater than 0";
    CHECK_LE(negative_swap_percentage_, 99) << "Swap percentage should be less than 100";
    for (int i = 0; i < max_buffer_size_; ++i) {
      buffer_ids_.push_back(i);
    }
  }

  // Initialize DB
  switch (this->layer_param_.video_shots_data_param().backend()) {
  case VideoShotsDataParameter_DB_LEVELDB:
    {
      leveldb::DB* db_temp;
      leveldb::Options options = GetLevelDBOptions();
      options.create_if_missing = false;
      LOG(INFO) << "Opening leveldb " << this->layer_param_.video_shots_data_param().source();
      leveldb::Status status = leveldb::DB::Open(
          options, this->layer_param_.video_shots_data_param().source(), &db_temp);
      CHECK(status.ok()) << "Failed to open leveldb "
                         << this->layer_param_.video_shots_data_param().source() << std::endl
                         << status.ToString();
      db_.reset(db_temp);
      iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
      iter_->SeekToFirst();
    }

    // Negative data buffer if needed
    if (this->layer_param_.video_shots_data_param().negative_dataset() != "") {
      leveldb::DB* db_temp;
      leveldb::Options options = GetLevelDBOptions();
      options.create_if_missing = false;
      LOG(INFO) << "Opening leveldb " << this->layer_param_.video_shots_data_param().negative_dataset();
      leveldb::Status status = leveldb::DB::Open(
          options, this->layer_param_.video_shots_data_param().negative_dataset(), &db_temp);
      CHECK(status.ok()) << "Failed to open leveldb "
                         << this->layer_param_.video_shots_data_param().negative_dataset()
                         << std::endl
                         << status.ToString();
      db_neg_.reset(db_temp);
      iter_neg_.reset(db_neg_->NewIterator(leveldb::ReadOptions()));
      iter_neg_->SeekToFirst();
    }
    break;
  case VideoShotsDataParameter_DB_LMDB:
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.video_shots_data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.video_shots_data_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";

    // Negative buffer if provided
    if (this->layer_param_.video_shots_data_param().negative_dataset() != "") {
      CHECK_EQ(mdb_env_create(&mdb_env_neg_), MDB_SUCCESS) << "mdb_env_create failed";
      CHECK_EQ(mdb_env_set_mapsize(mdb_env_neg_, 1099511627776), MDB_SUCCESS);  // 1TB
      CHECK_EQ(mdb_env_open(mdb_env_neg_,
               this->layer_param_.video_shots_data_param().negative_dataset().c_str(),
               MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
      CHECK_EQ(mdb_txn_begin(mdb_env_neg_, NULL, MDB_RDONLY, &mdb_txn_neg_), MDB_SUCCESS)
          << "mdb_txn_begin failed";
      CHECK_EQ(mdb_open(mdb_txn_neg_, NULL, 0, &mdb_dbi_neg_), MDB_SUCCESS)
          << "mdb_open failed";
      CHECK_EQ(mdb_cursor_open(mdb_txn_neg_, mdb_dbi_neg_, &mdb_cursor_neg_), MDB_SUCCESS)
          << "mdb_cursor_open failed";
      LOG(INFO) << "Opening lmdb " << this->layer_param_.video_shots_data_param().negative_dataset();
      CHECK_EQ(mdb_cursor_get(mdb_cursor_neg_, &mdb_key_neg_, &mdb_value_neg_, MDB_FIRST),
          MDB_SUCCESS) << "mdb_cursor_get failed";
    }
    break;
  }

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.video_shots_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.video_shots_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      switch (this->layer_param_.video_shots_data_param().backend()) {
      case VideoShotsDataParameter_DB_LEVELDB:
        iter_->Next();
        if (!iter_->Valid()) {
          iter_->SeekToFirst();
        }
        break;
      case VideoShotsDataParameter_DB_LMDB:
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
  VideoShots video_shots;
  switch (this->layer_param_.video_shots_data_param().backend()) {
  case VideoShotsDataParameter_DB_LEVELDB:
    video_shots.ParseFromString(iter_->value().ToString());
    break;
  case VideoShotsDataParameter_DB_LMDB:
    video_shots.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  //LOG(INFO) << video_shots.DebugString();

  // We expect one-d features only
  feature_size_ = video_shots.shot_words(0).float_data_size();
  if (this->layer_param_.video_shots_data_param().context_type() == VideoShotsDataParameter_CONTEXT_PAIRWISE) {
    context_size_ = 1;
  } else {
    context_size_ = this->layer_param_.video_shots_data_param().context_size();
  }
  
  CHECK_GE(feature_size_, 1);
  CHECK_GE(context_size_, 1);
  batch_size_ = this->layer_param_.video_shots_data_param().batch_size();
  CHECK_GE(batch_size_, 1);
  target_ctr_ = -1;
  context_ctr_ = 0;

  // Initialize the negative added vector
  neg_added_from_same_video_.insert(neg_added_from_same_video_.begin(), batch_size_, 0);

  // Initialize video-ids vector
  video_ids_.insert(video_ids_.begin(), batch_size_, 0);

  // We use a small hack: channels = context_size_+1, height = feature_size_
  (*top)[0]->Reshape(
      this->layer_param_.video_shots_data_param().batch_size(),
        context_size_+1+num_negative_samples_, // context-features + target featuers + negatives
        feature_size_, 1);
  this->prefetch_data_.Reshape(this->layer_param_.video_shots_data_param().batch_size(),
      context_size_+1+num_negative_samples_, feature_size_, 1);

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();

  // Video id if needed
  if (this->output_labels_) {
    (*top)[1]->Reshape(this->layer_param_.video_shots_data_param().batch_size(), 1, 1, 1);
    this->prefetch_label_.Reshape(this->layer_param_.video_shots_data_param().batch_size(),
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

   for (int nid = 0; nid < (FLAGS_max_tries*max_buffer_size_) ; ++nid) {
    if (num_negatives_added % 1000 == 0) {
      LOG(INFO) << "Added " << num_negatives_added << " negatives to buffer";
    }

    switch (this->layer_param_.video_shots_data_param().backend()) {
      case VideoShotsDataParameter_DB_LEVELDB:

        if (this->layer_param_.video_shots_data_param().negative_dataset() != "") {
          CHECK(iter_neg_);
          CHECK(iter_neg_->Valid());
          video_shots.ParseFromString(iter_neg_->value().ToString());
          iter_neg_->Next();
          if (!iter_neg_->Valid()) {
            iter_neg_->SeekToFirst();
          }
        } else {
          CHECK(iter_);
          CHECK(iter_->Valid());
          video_shots.ParseFromString(iter_->value().ToString());
          iter_->Next();
          if (!iter_->Valid()) {
            iter_->SeekToFirst();
          }
        }
        break;

      case VideoShotsDataParameter_DB_LMDB:
        if (this->layer_param_.video_shots_data_param().negative_dataset() != "") {

          CHECK_EQ(mdb_cursor_get(mdb_cursor_neg_, &mdb_key_neg_,
                &mdb_value_neg_, MDB_GET_CURRENT), MDB_SUCCESS);
          video_shots.ParseFromArray(mdb_value_neg_.mv_data,
            mdb_value_neg_.mv_size);
          if (mdb_cursor_get(mdb_cursor_neg_, &mdb_key_neg_, &mdb_value_neg_, MDB_NEXT)
              != MDB_SUCCESS) {
            CHECK_EQ(mdb_cursor_get(mdb_cursor_neg_, &mdb_key_neg_, &mdb_value_neg_,
                     MDB_FIRST), MDB_SUCCESS);
          }
        } else {
          CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
          video_shots.ParseFromArray(mdb_value_.mv_data,
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

    int num_shots = video_shots.shot_words_size();


    if (this->layer_param_.video_shots_data_param().negative_dataset() == "") {

      int sample_shot = rand() % num_shots;

      string negative_key = stringprintf("%d:%d", video_shots.video_id(),
          video_shots.shot_ids(sample_shot));
      if (negative_keys_set_.find(negative_key) == negative_keys_set_.end()) {
        // Copy data into negative samples
        for (int f = 0; f < feature_size_; ++f) {
          negatives_mutable_data[num_negatives_added*feature_size_ + f] =
            video_shots.shot_words(sample_shot).float_data(f);
        }
        negative_id_to_key_.push_back(negative_key);
        negative_keys_set_.insert(negative_key);
        num_negatives_added++;
      }

    } else {
      for (int sample_shot = 0; sample_shot < num_shots; ++sample_shot) {
        string negative_key = stringprintf("%d:%d", video_shots.video_id(),
            video_shots.shot_ids(sample_shot));
        if (negative_keys_set_.find(negative_key) == negative_keys_set_.end()) {
          // Copy data into negative samples
          for (int f = 0; f < feature_size_; ++f) {
            negatives_mutable_data[num_negatives_added*feature_size_ + f] =
              video_shots.shot_words(sample_shot).float_data(f);
          }
          negative_id_to_key_.push_back(negative_key);
          negative_keys_set_.insert(negative_key);
          num_negatives_added++;
        }
      }
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
  if (this->layer_param_.video_shots_data_param().negative_dataset() != "") {
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

template <typename Dtype>
void VideoShotsDataLayer<Dtype>::InsertIntoQueue(const VideoShots& video_shots,
                     Dtype* top_data,
                     int &item_id) {

  int context_id = 0, start_j = 0;
  int half_context_size = context_size_/2;
  std::vector<int> video_index_negs;
  int num_same_video_negs = 0;
  switch (this->layer_param_.video_shots_data_param().context_type()) {

    // -------------------- Pairwise only ----------------------
    case VideoShotsDataParameter_CONTEXT_PAIRWISE:
      for (int i = this->target_ctr_; i < video_shots.shot_words_size(); ++i) {
        if (i==this->target_ctr_) {
          start_j = this->context_ctr_;
        } else {
          start_j = 0;
        }
        for (int j = start_j; j < video_shots.shot_words_size(); ++j) {
          if (i==j) {
            continue;
          }
          //LOG(INFO) << "Adding: "<< item_id << ":" << video_shots.video_id() << ":"  << i << ":" << j;
          // The first channel contains the actual target word
          for (int feature_id = 0; feature_id < this->datum_height_; ++feature_id) {
            top_data[item_id*this->datum_channels_*this->datum_height_ + feature_id] =
                 video_shots.shot_words(i).float_data(feature_id);
            top_data[(item_id*this->datum_channels_ + 1) * this->datum_height_ + feature_id] =
                 video_shots.shot_words(j).float_data(feature_id);
          }

          if (this->output_shot_distance_) {
            //video_ids_[item_id] = Dtype(1.0/ Dtype(this->layer_param_.video_shots_data_param().max_shot_distance() + abs(i-j)));
            //if (abs(i-j) >= this->layer_param_.video_shots_data_param().max_shot_distance()) {
            //  video_ids_[item_id] = Dtype(1.0/Dtype(this->layer_param_.video_shots_data_param().max_shot_distance() * 2));
            //}
            video_ids_[item_id] = Dtype(abs(i-j));
            if (abs(i-j) >= this->layer_param_.video_shots_data_param().max_shot_distance()) {
              video_ids_[item_id] = Dtype(this->layer_param_.video_shots_data_param().max_shot_distance());
            }
          } else {
            video_ids_[item_id] = video_shots.video_id();
          }
          this->target_ctr_ = i;
          this->context_ctr_ = j;
          item_id++;
          if (item_id >= batch_size_) {
            break;
          }
        }

        if (item_id >= batch_size_) {
          break;
        }
      }


      if (item_id != batch_size_) {
        this->target_ctr_ = -1;
      } else {
        if (this->context_ctr_ == (video_shots.shot_words_size()-1)) {
          this->context_ctr_ = 0;
          this->target_ctr_++;
          if (this->target_ctr_ >= (video_shots.shot_words_size())) {
            this->target_ctr_ = -1;
          }
        } else {
          this->context_ctr_++;
        }
      }

      break;

    // ------------------- Context around the target ---------------------
    case VideoShotsDataParameter_CONTEXT_WINDOW:
      CHECK(context_size_%2 == 0) << "Context size should be even in this setting!";
      // For negatives
      video_index_negs.clear();
      for (int nid = 0; nid < video_shots.shot_words_size(); ++nid) {
        video_index_negs.push_back(nid);
      }

      for (int i = this->target_ctr_; i < video_shots.shot_words_size(); ++i) {

        // Avoid border cases
        /*if ( (i - half_context_size) < 0 || (i + half_context_size) >= video_shots.shot_words_size()) {
          context_id = 0;
          this->target_ctr_++;
        }*/

        for (int feature_id = 0; feature_id < this->datum_height_; ++feature_id) {
          top_data[item_id*this->datum_channels_*this->datum_height_ + feature_id] =
                   video_shots.shot_words(i).float_data(feature_id);
        }

        /*LOG(INFO) << "Top target --------------------------------> "
                                          << ":" << top_data[item_id*this->datum_channels_*this->datum_height_ + 1]
                                          << ":" << top_data[item_id*this->datum_channels_*this->datum_height_ + 10]
                                          << ":" << top_data[item_id*this->datum_channels_*this->datum_height_+ 100]
                                          << ":" << top_data[item_id*this->datum_channels_*this->datum_height_ + 500]
                                          << ":" << top_data[item_id*this->datum_channels_*this->datum_height_ + 2000];

        */
        context_id = 0;
        for (int j = i-half_context_size; j <= (i+half_context_size); ++j) {
          if (i==j) {
            continue;
          }
          if ( (j<0) || (j >= video_shots.shot_words_size())) {
            // Represent end-tokens by a dummy feature, where last feature-id = 1
            for (int feature_id = 0; feature_id < (this->datum_height_-1); ++feature_id) {
              top_data[(item_id*this->datum_channels_ + context_id + 1) * this->datum_height_
                + feature_id] = 0;
            }
            top_data[(item_id*this->datum_channels_ + context_id + 1) * this->datum_height_
                + (this->datum_height_-1)] = 1;
          } else {
         
            for (int feature_id = 0; feature_id < this->datum_height_; ++feature_id) {
              top_data[(item_id*this->datum_channels_ + context_id + 1)*this->datum_height_
                + feature_id] =
                   video_shots.shot_words(j).float_data(feature_id);
            } 
          }
          context_id++;
        }
        this->target_ctr_++;
        video_ids_[item_id] = video_shots.video_id();

        // Add same video-negatives
        num_same_video_negs = 0;
        if (num_negative_samples_ > 0) {
          std::random_shuffle(video_index_negs.begin(), video_index_negs.end());
          for (int nid = 0; (nid < video_shots.shot_words_size())
                           && (num_same_video_negs < max_same_video_negs_); ++nid) {
            if (video_index_negs[nid] == i) {
              continue;
            }
            for (int feature_id = 0; feature_id < (this->datum_height_-1); ++feature_id) {
                          top_data[(item_id*this->datum_channels_ + this->context_size_ + 1 + num_same_video_negs) * this->datum_height_ + feature_id] =
                            video_shots.shot_words(video_index_negs[nid]).float_data(feature_id);

            }
            num_same_video_negs++;
          }
          neg_added_from_same_video_[item_id] = num_same_video_negs;
        }


        item_id++;
        if (item_id >= batch_size_) {
          break;
        }
      }

      if (this->target_ctr_ >= (video_shots.shot_words_size())) {
        this->target_ctr_ = -1;
      }

      break;

    // -------------------- Context right before the shot -----------------
    case VideoShotsDataParameter_CONTEXT_PAST:

      for (int i = this->target_ctr_; i < video_shots.shot_words_size(); ++i) {
        for (int feature_id = 0; feature_id < this->datum_height_; ++feature_id) {
          top_data[item_id*this->datum_channels_*this->datum_height_ + feature_id] =
                   video_shots.shot_words(i).float_data(feature_id);
        }

        context_id = 0;
        for (int j = i-context_size_; j < i; ++j) {
          if ( j<0 ) {
            // Represent end-tokens by a dummy feature, where last feature-id = 1
            for (int feature_id = 0; feature_id < (this->datum_height_-1); ++feature_id) {
              top_data[(item_id*this->datum_channels_ + context_id + 1) * this->datum_height_
                + feature_id] = 0;
            }
            top_data[(item_id*this->datum_channels_ + context_id + 1) * this->datum_height_
                + (this->datum_height_-1)] = 1;
          } else {
         
            for (int feature_id = 0; feature_id < this->datum_height_; ++feature_id) {
              top_data[(item_id*this->datum_channels_ + context_id + 1)*this->datum_height_
                + feature_id] =
                   video_shots.shot_words(j).float_data(feature_id);
            } 
          }
          context_id++;
        }
        this->target_ctr_++;
        video_ids_[item_id] = video_shots.video_id();
        item_id++;
        if (item_id >= batch_size_) {
          break;
        }
      }


      if (this->target_ctr_ >= (video_shots.shot_words_size())) {
        this->target_ctr_ = -1;
      }
      break;

    default:
      LOG(FATAL) << "Unknown context type";
  }

}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void VideoShotsDataLayer<Dtype>::InternalThreadEntry() {
  VideoShots video_shots;
  CHECK(this->prefetch_data_.count());

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }

  int item_id = 0;

  //LOG(INFO) << "Starting to insert old video: ";


  if (this->target_ctr_ >= 0) {
    InsertIntoQueue(current_video_shots_, top_data, item_id);
  }
  //LOG(INFO) << "Have inseted: " << item_id << " elements ...";
  while (item_id < batch_size_) {
    
    VideoShots video_shots;
    // get a blob
    switch (this->layer_param_.video_shots_data_param().backend()) {
      case VideoShotsDataParameter_DB_LEVELDB:
        CHECK(iter_);
        CHECK(iter_->Valid());
        video_shots.ParseFromString(iter_->value().ToString());
        break;
      case VideoShotsDataParameter_DB_LMDB:
        // Perhaps this is unecessay, we could directly take mdb_value_
        CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
        video_shots.ParseFromArray(mdb_value_.mv_data,
            mdb_value_.mv_size);
        break;
      default:
        LOG(FATAL) << "Unknown database backend";
    }

    current_video_shots_ = video_shots;
    // Check to ensure that the video shots is valid
    CHECK(video_shots.has_video_id()) << "No video id found for video shot";
    CHECK_GE(video_shots.shot_words_size(), 1) << "No shot word found: "
                                              << video_shots.video_id();

    CHECK_EQ(video_shots.shot_words(0).float_data_size(), this->datum_height_)
      << "Wrong feature size: " << video_shots.video_id();

    this->target_ctr_ = 0;
    this->context_ctr_ = 0;

    // Main insertion into top-data
    //LOG(INFO) << "Trying to insert video-id: " << video_shots.video_id();
    //LOG(INFO) << "Done ...";
    InsertIntoQueue(video_shots, top_data, item_id);

    // go to the next iter
    switch (this->layer_param_.video_shots_data_param().backend()) {
      case VideoShotsDataParameter_DB_LEVELDB:
        iter_->Next();
        if (!iter_->Valid()) {
          // We have reached the end. Restart from the first.
          LOG(INFO) << "Restarting data prefetching from start.";
          iter_->SeekToFirst();
        }
        break;
      case VideoShotsDataParameter_DB_LMDB:
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

    // ------------------ Adding Negative to buffer
    // Add the word to negative buffer
    if (num_negative_samples_ > 0) {
      // Swap values in buffer if needed
      if (negative_swap_percentage_ > 0) {
        for (int j = 0; j < video_shots.shot_ids_size(); ++j) {
          string new_negative_key = stringprintf("%d:%d", video_shots.video_id(),
              video_shots.shot_ids(j));
          // Replace only if the shot is already not in the negative buffer
          if (negative_keys_set_.find(new_negative_key) == negative_keys_set_.end()) {
            int negative_pos = AddToBuffer(video_shots.shot_words(j));
            if (negative_pos >= 0) {
              string old_negative_key = negative_id_to_key_[negative_pos];
              negative_id_to_key_[negative_pos] = new_negative_key;
              negative_keys_set_.erase(old_negative_key);
              negative_keys_set_.insert(new_negative_key);
            }
          } // if not already in buffer
        } // loop through shots
      } // percnetage > 0
    } // num_negative_samples_ > 0
    // --------------------- End of negative buffer addition
  }

  // Add negative samples
  for (int b = 0; b < batch_size_; ++b) {  
    if (num_negative_samples_ > 0) {
      const Dtype* negatives_data = negatives_.cpu_data();
      // Sample the remaining from negatives
      RandomShuffleTopids(num_negative_samples_);
      for (int negative_id = (this->context_size_ + 1 + neg_added_from_same_video_[b]);
          negative_id < (1 + this->context_size_ + num_negative_samples_); ++negative_id) {
        int neg_id = static_cast<int>(this->buffer_ids_[negative_id - context_size_ - 1]);
        for (int feature_id = 0; feature_id < feature_size_; ++feature_id) {

          top_data[(b*this->datum_channels_ + negative_id) * this->datum_height_ + feature_id] =
           negatives_data[neg_id*feature_size_ + feature_id];
        }
        /* LOG(INFO) << "neg-id: " << neg_id << ":" << negatives_data[neg_id*feature_size_ + 1]
                                          << ":" << negatives_data[neg_id*feature_size_ + 10]
                                          << ":" << negatives_data[neg_id*feature_size_ + 100]
                                          << ":" << negatives_data[neg_id*feature_size_ + 500]
                                          << ":" << negatives_data[neg_id*feature_size_ + 2000];
        */

      }
    }

    if (this->output_labels_) {
      top_label[b] = video_ids_[b];
    }

    /*string debug_str = stringprintf("%d:%f,%f,%f,%f,%f",static_cast<int>(top_label[b]),
                                                        static_cast<float>(top_data[(b*this->datum_channels_ + 0)*this->datum_height_ + 0]),
                                                        static_cast<float>(top_data[(b*this->datum_channels_ + 1)*this->datum_height_ + 0]),
                                                        static_cast<float>(top_data[(b*this->datum_channels_ + 2)*this->datum_height_ + 0]),
                                                        static_cast<float>(top_data[(b*this->datum_channels_ + 7)*this->datum_height_ + 0]),
                                                        static_cast<float>(top_data[(b*this->datum_channels_ + 11)*this->datum_height_ + 0]));
    LOG(INFO) << "BATCH-INFO: " << debug_str;*/
  }

  //LOG(INFO) << "Video shot .... inserted";

}

INSTANTIATE_CLASS(VideoShotsDataLayer);

}  // namespace caffe
