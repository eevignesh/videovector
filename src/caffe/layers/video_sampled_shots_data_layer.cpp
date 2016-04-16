#include <leveldb/db.h>
#include <stdint.h>
#include <algorithm>
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
DEFINE_int32(max_tries_for_negs, 100, "Maximum number of tries for adding negatives");

namespace caffe {

template <typename Dtype>
inline int VideoSampledShotsDataLayer<Dtype>::AddToBuffer(const Datum datum) {
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
inline void VideoSampledShotsDataLayer<Dtype>::RandomShuffleTopids(int n) {
    random_unique(buffer_ids_.begin(), buffer_ids_.end(), n);
  }

  template <typename Dtype>
  VideoSampledShotsDataLayer<Dtype>::~VideoSampledShotsDataLayer<Dtype>() {
    this->JoinPrefetchThread();
    // clean up the database resources
    switch (this->layer_param_.video_sampled_shots_data_param().backend()) {
    case VideoSampledShotsDataParameter_DB_LEVELDB:
      break;  // do nothing
    case VideoSampledShotsDataParameter_DB_LMDB:
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
  void VideoSampledShotsDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
        vector<Blob<Dtype>*>* top) {

    // Set up the negative buffer
    num_negative_samples_ = this->layer_param_.video_sampled_shots_data_param().num_negative_samples();

    output_shot_distance_ = this->layer_param_.video_sampled_shots_data_param().output_shot_distance();

    max_buffer_size_ = 0;
    max_same_video_negs_ = this->layer_param_.video_sampled_shots_data_param().max_same_video_negs();

    if (num_negative_samples_ > 0) {
      max_buffer_size_ = this->layer_param_.video_sampled_shots_data_param().max_buffer_size();
      negative_swap_percentage_ = this->layer_param_.video_sampled_shots_data_param().negative_swap_percentage();
      CHECK_GE(negative_swap_percentage_, 0) << "Swap percentage should be greater than 0";
      CHECK_LE(negative_swap_percentage_, 99) << "Swap percentage should be less than 100";
      for (int i = 0; i < max_buffer_size_; ++i) {
        buffer_ids_.push_back(i);
      }
    }

    // Initialize DB
    switch (this->layer_param_.video_sampled_shots_data_param().backend()) {
    case VideoSampledShotsDataParameter_DB_LEVELDB:
      {
        leveldb::DB* db_temp;
        leveldb::Options options = GetLevelDBOptions();
        options.create_if_missing = false;
        LOG(INFO) << "Opening leveldb " << this->layer_param_.video_sampled_shots_data_param().source();
        leveldb::Status status = leveldb::DB::Open(
            options, this->layer_param_.video_sampled_shots_data_param().source(), &db_temp);
        CHECK(status.ok()) << "Failed to open leveldb "
                           << this->layer_param_.video_sampled_shots_data_param().source() << std::endl
                           << status.ToString();
        db_.reset(db_temp);
        iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
        iter_->SeekToFirst();
      }

      // Negative data buffer if needed
      if (this->layer_param_.video_sampled_shots_data_param().negative_dataset() != "") {
        leveldb::DB* db_temp;
        leveldb::Options options = GetLevelDBOptions();
        options.create_if_missing = false;
        LOG(INFO) << "Opening leveldb " << this->layer_param_.video_sampled_shots_data_param().negative_dataset();
        leveldb::Status status = leveldb::DB::Open(
            options, this->layer_param_.video_sampled_shots_data_param().negative_dataset(), &db_temp);
        CHECK(status.ok()) << "Failed to open leveldb "
                           << this->layer_param_.video_sampled_shots_data_param().negative_dataset()
                           << std::endl
                           << status.ToString();
        db_neg_.reset(db_temp);
        iter_neg_.reset(db_neg_->NewIterator(leveldb::ReadOptions()));
        iter_neg_->SeekToFirst();
      }
      break;
    case VideoSampledShotsDataParameter_DB_LMDB:
      CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
      CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
      CHECK_EQ(mdb_env_open(mdb_env_,
               this->layer_param_.video_sampled_shots_data_param().source().c_str(),
               MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
      CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
          << "mdb_txn_begin failed";
      CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
          << "mdb_open failed";
      CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
          << "mdb_cursor_open failed";
      LOG(INFO) << "Opening lmdb " << this->layer_param_.video_sampled_shots_data_param().source();
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
          MDB_SUCCESS) << "mdb_cursor_get failed";

      // Negative buffer if provided
      if (this->layer_param_.video_sampled_shots_data_param().negative_dataset() != "") {
        CHECK_EQ(mdb_env_create(&mdb_env_neg_), MDB_SUCCESS) << "mdb_env_create failed";
        CHECK_EQ(mdb_env_set_mapsize(mdb_env_neg_, 1099511627776), MDB_SUCCESS);  // 1TB
        CHECK_EQ(mdb_env_open(mdb_env_neg_,
                 this->layer_param_.video_sampled_shots_data_param().negative_dataset().c_str(),
                 MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
        CHECK_EQ(mdb_txn_begin(mdb_env_neg_, NULL, MDB_RDONLY, &mdb_txn_neg_), MDB_SUCCESS)
            << "mdb_txn_begin failed";
        CHECK_EQ(mdb_open(mdb_txn_neg_, NULL, 0, &mdb_dbi_neg_), MDB_SUCCESS)
            << "mdb_open failed";
        CHECK_EQ(mdb_cursor_open(mdb_txn_neg_, mdb_dbi_neg_, &mdb_cursor_neg_), MDB_SUCCESS)
            << "mdb_cursor_open failed";
        LOG(INFO) << "Opening lmdb " << this->layer_param_.video_sampled_shots_data_param().negative_dataset();
        CHECK_EQ(mdb_cursor_get(mdb_cursor_neg_, &mdb_key_neg_, &mdb_value_neg_, MDB_FIRST),
            MDB_SUCCESS) << "mdb_cursor_get failed";
      }
      break;
    }

    // Check if we would need to randomly skip a few data points
    if (this->layer_param_.video_sampled_shots_data_param().rand_skip()) {
      unsigned int skip = caffe_rng_rand() %
                          this->layer_param_.video_sampled_shots_data_param().rand_skip();
      LOG(INFO) << "Skipping first " << skip << " data points.";
      while (skip-- > 0) {
        switch (this->layer_param_.video_sampled_shots_data_param().backend()) {
        case VideoSampledShotsDataParameter_DB_LEVELDB:
          iter_->Next();
          if (!iter_->Valid()) {
            iter_->SeekToFirst();
          }
          break;
        case VideoSampledShotsDataParameter_DB_LMDB:
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
    switch (this->layer_param_.video_sampled_shots_data_param().backend()) {
    case VideoSampledShotsDataParameter_DB_LEVELDB:
      video_shots.ParseFromString(iter_->value().ToString());
      break;
    case VideoSampledShotsDataParameter_DB_LMDB:
      video_shots.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }

    //LOG(INFO) << video_shots.DebugString();

    // We expect one-d features only
    feature_size_ = video_shots.shot_words(0).float_data_size();
    if (this->layer_param_.video_sampled_shots_data_param().context_type() == VideoSampledShotsDataParameter_CONTEXT_PAIRWISE) {
      context_size_ = 2;
    } else {
      context_size_ = this->layer_param_.video_sampled_shots_data_param().context_size();
    }
    
    CHECK_GE(feature_size_, 1);
    CHECK_GE(context_size_, 2);
    batch_size_ = this->layer_param_.video_sampled_shots_data_param().batch_size();
    CHECK_GE(batch_size_, 1);

    // Initialize the negative added vector
    neg_added_from_same_video_.insert(neg_added_from_same_video_.begin(), batch_size_, 0);

    // We use a small hack: channels = context_size_+1, height = feature_size_
    (*top)[0]->Reshape(
        this->layer_param_.video_sampled_shots_data_param().batch_size(),
          context_size_+num_negative_samples_, // context-features + target featuers + negatives
          feature_size_, 1);
    this->prefetch_data_.Reshape(this->layer_param_.video_sampled_shots_data_param().batch_size(),
        context_size_+num_negative_samples_, feature_size_, 1);

    LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
        << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
        << (*top)[0]->width();

    // Video id if needed
    if (this->output_labels_) {
      (*top)[1]->Reshape(this->layer_param_.video_sampled_shots_data_param().batch_size(), 1, 1, 1);
      this->prefetch_label_.Reshape(this->layer_param_.video_sampled_shots_data_param().batch_size(),
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

     for (int nid = 0; nid < (FLAGS_max_tries_for_negs*max_buffer_size_) ; ++nid) {
      if (num_negatives_added % 1000 == 0) {
        LOG(INFO) << "Added " << num_negatives_added << " negatives to buffer";
      }

      switch (this->layer_param_.video_sampled_shots_data_param().backend()) {
        case VideoSampledShotsDataParameter_DB_LEVELDB:

          if (this->layer_param_.video_sampled_shots_data_param().negative_dataset() != "") {
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

        case VideoSampledShotsDataParameter_DB_LMDB:
          if (this->layer_param_.video_sampled_shots_data_param().negative_dataset() != "") {

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


      if (this->layer_param_.video_sampled_shots_data_param().negative_dataset() == "") {

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
    this->datum_channels_ = context_size_ + num_negative_samples_;
    this->datum_height_ = feature_size_;
    this->datum_width_ = 1;
    this->datum_size_ = feature_size_ * this->datum_channels_;

    // Close the negative dataset buffer
    if (this->layer_param_.video_sampled_shots_data_param().negative_dataset() != "") {
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
  void VideoSampledShotsDataLayer<Dtype>::AddSamplesToTop(const VideoShots& video_shots,
                     Dtype* top_data,
                     const int& item_id,
                     int& added_negatives,
                     int& video_id,
                     bool& video_added) {
  video_id = -1;
  added_negatives = 0;
  video_added = false;
  int half_context_size = context_size_/2;
  int context_id = 0, begin_frame = 0;
  int frame_id = 0, max_sample_length = 0, sample_length = 0;
  string ned_id_string = "", context_id_string = "";

  // If only one frame ... continue
  if (video_shots.shot_words_size() < 2) {
    return;
  }
  vector<int> rand_perm_ids(video_shots.shot_words_size());
  std::iota(rand_perm_ids.begin(), rand_perm_ids.end(), 0);

  switch (this->layer_param_.video_sampled_shots_data_param().context_type()) {

    // -------------------- Pairwise only ----------------------
    case VideoSampledShotsDataParameter_CONTEXT_PAIRWISE:
      random_unique(rand_perm_ids.begin(), rand_perm_ids.end(), 2);

      // The first channel contains the actual target word
      for (int feature_id = 0; feature_id < this->datum_height_; ++feature_id) {
        top_data[item_id*this->datum_channels_*this->datum_height_ + feature_id] =
             video_shots.shot_words(rand_perm_ids[0]).float_data(feature_id);
        top_data[(item_id*this->datum_channels_ + 1) * this->datum_height_ + feature_id] =
             video_shots.shot_words(rand_perm_ids[1]).float_data(feature_id);
      }

      if (this->output_shot_distance_) {
        //video_ids_[item_id] = Dtype(1.0/ Dtype(this->layer_param_.video_sampled_shots_data_param().max_shot_distance() + abs(i-j)));
        //if (abs(i-j) >= this->layer_param_.video_sampled_shots_data_param().max_shot_distance()) {
        //  video_ids_[item_id] = Dtype(1.0/Dtype(this->layer_param_.video_sampled_shots_data_param().max_shot_distance() * 2));
        //}
        video_id = Dtype(abs(rand_perm_ids[0]-rand_perm_ids[1]));
        if (abs(rand_perm_ids[0]-rand_perm_ids[1]) >= this->layer_param_.video_sampled_shots_data_param().max_shot_distance()) {
          video_id = Dtype(this->layer_param_.video_sampled_shots_data_param().max_shot_distance());
        }
      } else {
        video_id = video_shots.video_id();
      }

      video_added = true;

      break;

    // ------------------- Context around the target ---------------------
    case VideoSampledShotsDataParameter_CONTEXT_WINDOW:

      if (rand_perm_ids.size() < context_size_) {
        return;
      }

      // Random shuffle
      random_unique(rand_perm_ids.begin(), rand_perm_ids.end(), context_size_);

      CHECK(context_size_%2 == 1) << "Context size should be even in this setting!";

      // Sort the top context_size ids
      std::sort(rand_perm_ids.begin(), rand_perm_ids.begin() + context_size_);

      for (int i = 0; i < context_size_; ++i) {

        if (i == (half_context_size)) {
          for (int feature_id = 0; feature_id < this->datum_height_; ++feature_id) {
            top_data[item_id*this->datum_channels_*this->datum_height_ + feature_id] =
                     video_shots.shot_words(rand_perm_ids[i]).float_data(feature_id);
          }
        } else {
          for (int feature_id = 0; feature_id < this->datum_height_; ++feature_id) {
            top_data[(item_id*this->datum_channels_ + context_id + 1)*this->datum_height_ + feature_id] =
                     video_shots.shot_words(rand_perm_ids[i]).float_data(feature_id);
          }
          context_id++;
        }
      }

      CHECK_EQ(context_id, context_size_-1);


      /*LOG(INFO) << "Sampled context: " << rand_perm_ids[0]
                << ":" << rand_perm_ids[1]
                << ":" << rand_perm_ids[2]
                << ":" << rand_perm_ids[3]
                << ":" << rand_perm_ids[4];*/

      // Set the video-id
      video_id = video_shots.video_id();
      
      video_added = true;
      /*LOG(INFO) << "Top target --------------------------------> "
        << ":" << top_data[item_id*this->datum_channels_*this->datum_height_ + 1]
        << ":" << top_data[item_id*this->datum_channels_*this->datum_height_ + 10]
        << ":" << top_data[item_id*this->datum_channels_*this->datum_height_+ 100]
        << ":" << top_data[item_id*this->datum_channels_*this->datum_height_ + 500]
        << ":" << top_data[item_id*this->datum_channels_*this->datum_height_ + 2000];
      */
      
      //LOG(INFO) << "NEGINFO: " << context_size_ << ":" << video_shots.shot_words_size() << ":" << added_negatives << ":" << max_same_video_negs_;

      // Add same video-negatives
      if ((num_negative_samples_ > 0) &&
          (video_shots.shot_words_size() > context_size_)) {

        std::random_shuffle(rand_perm_ids.begin() + context_size_, rand_perm_ids.end());

        for (int nid = context_size_; (nid < video_shots.shot_words_size())
                         && (added_negatives < max_same_video_negs_); ++nid) {

          //LOG(INFO) << "NEG: " << rand_perm_ids[nid] <<  " : " << rand_perm_ids[half_context_size-1]
          //  << " : " << rand_perm_ids[half_context_size+1];
          if ((rand_perm_ids[nid] < rand_perm_ids[half_context_size-1]) || 
                   (rand_perm_ids[nid] > rand_perm_ids[half_context_size+1])) {

            for (int feature_id = 0; feature_id < (this->datum_height_-1); ++feature_id) {
              top_data[(item_id*this->datum_channels_ + this->context_size_ + added_negatives)
                        * this->datum_height_ + feature_id] =
                video_shots.shot_words(rand_perm_ids[nid]).float_data(feature_id);

            }

            //ned_id_string += stringprintf(":%d", rand_perm_ids[nid]);
            added_negatives++;
          }
        }
      }

      //LOG(INFO) << "Sampled negs: " << ned_id_string << " added-negs: " << added_negatives;

      break;

    // -------------------- Context right before the shot -----------------
    case VideoSampledShotsDataParameter_CONTEXT_PAST:

      if (rand_perm_ids.size() < context_size_) {
        return;
      }

      // Random shuffle
      random_unique(rand_perm_ids.begin(), rand_perm_ids.end(), context_size_);

      CHECK_GE(context_size_, 2);

      // Sort the top context_size ids
      std::sort(rand_perm_ids.begin(), rand_perm_ids.begin() + context_size_);
      
      context_id = 0;
      for (int i = 0; i < context_size_; ++i) {

        if (i == (context_size_-1)) {
          for (int feature_id = 0; feature_id < this->datum_height_; ++feature_id) {
            top_data[item_id*this->datum_channels_*this->datum_height_ + feature_id] =
                     video_shots.shot_words(rand_perm_ids[i]).float_data(feature_id);
          }
        } else {
          for (int feature_id = 0; feature_id < this->datum_height_; ++feature_id) {
            top_data[(item_id*this->datum_channels_ + context_id + 1)*this->datum_height_ + feature_id] =
                     video_shots.shot_words(rand_perm_ids[i]).float_data(feature_id);
          }
          context_id_string += stringprintf(":%d", rand_perm_ids[i]);
          context_id++;
        }
      }

      CHECK_EQ(context_id, context_size_-1);

      // Set the video-id
      video_id = video_shots.video_id();
      
      video_added = true;
      /*LOG(INFO) << "Top target --------------------------------> "
        << ":" << top_data[item_id*this->datum_channels_*this->datum_height_ + 1]
        << ":" << top_data[item_id*this->datum_channels_*this->datum_height_ + 10]
        << ":" << top_data[item_id*this->datum_channels_*this->datum_height_+ 100]
        << ":" << top_data[item_id*this->datum_channels_*this->datum_height_ + 500]
        << ":" << top_data[item_id*this->datum_channels_*this->datum_height_ + 2000];
      */



      // Add same video-negatives
      //LOG(INFO) << "NEGINFO: " << context_size_ << ":" << video_shots.shot_words_size() << ":" << added_negatives << ":" << max_same_video_negs_;
      added_negatives = 0;
      if ((num_negative_samples_ > 0) &&
          (video_shots.shot_words_size() > context_size_)) {

        std::random_shuffle(rand_perm_ids.begin()+context_size_, rand_perm_ids.end());
        for (int nid = context_size_; (nid < video_shots.shot_words_size())
                         && (added_negatives < max_same_video_negs_); ++nid) {
//          if ((rand_perm_ids[nid] < rand_perm_ids[context_size_-2])) {
          if ((rand_perm_ids[nid] < rand_perm_ids[1])) {

            for (int feature_id = 0; feature_id < (this->datum_height_-1); ++feature_id) {
              top_data[(item_id*this->datum_channels_ + this->context_size_ + added_negatives)
                        * this->datum_height_ + feature_id] =
                video_shots.shot_words(rand_perm_ids[nid]).float_data(feature_id);

            }
            ned_id_string += stringprintf(":%d", rand_perm_ids[nid]);
            added_negatives++;
          }
        }
      }

      /*LOG(INFO) << "Sampled negs: N" << ned_id_string << "-- C" << context_id_string
        << " -- P " << rand_perm_ids[context_size_-1]
        << " added-negs: " << added_negatives;*/


      break;

    // -------------------- Context right before the shot -----------------
    case VideoSampledShotsDataParameter_CONTEXT_PAST_CONTINUOUS:

      if (rand_perm_ids.size() < context_size_) {
        return;
      }
      CHECK_GE(context_size_, 2);

      // Randomly pick a point in the set of frames
      //random_unique(rand_perm_ids.begin(), rand_perm_ids.begin() + (rand_perm_ids.size() - context_size + 1), 1);
      
      max_sample_length = (rand_perm_ids.size() - context_size_)/(context_size_-1);
      sample_length = (rand() % (max_sample_length+1));

      begin_frame = rand() % (rand_perm_ids.size() -
          (context_size_-1)*(sample_length) - context_size_ + 1) ;

     
      context_id = 0;
      for (int i = 0; i < context_size_; ++i) {
        frame_id = begin_frame + i*(sample_length + 1);
        if (i ==  (context_size_-1)) {
          for (int feature_id = 0; feature_id < this->datum_height_; ++feature_id) {
            top_data[item_id*this->datum_channels_*this->datum_height_ + feature_id] =
                     video_shots.shot_words(frame_id).float_data(feature_id);
          }
        } else {
          for (int feature_id = 0; feature_id < this->datum_height_; ++feature_id) {
            top_data[(item_id*this->datum_channels_ + context_id + 1)*this->datum_height_ + feature_id] =
                     video_shots.shot_words(frame_id).float_data(feature_id);
          }
          context_id_string += stringprintf(":%d", frame_id);
          context_id++;
        }
      }

      CHECK_EQ(context_id, context_size_-1);

      // Set the video-id
      video_id = video_shots.video_id();
      
      video_added = true;
      /*LOG(INFO) << "Top target --------------------------------> "
        << ":" << top_data[item_id*this->datum_channels_*this->datum_height_ + 1]
        << ":" << top_data[item_id*this->datum_channels_*this->datum_height_ + 10]
        << ":" << top_data[item_id*this->datum_channels_*this->datum_height_+ 100]
        << ":" << top_data[item_id*this->datum_channels_*this->datum_height_ + 500]
        << ":" << top_data[item_id*this->datum_channels_*this->datum_height_ + 2000];
      */



      // Add same video-negatives
      //LOG(INFO) << "NEGINFO: " << context_size_ << ":" << video_shots.shot_words_size() << ":" << added_negatives << ":" << max_same_video_negs_;
      added_negatives = 0;
      if ((num_negative_samples_ > 0) &&
          (begin_frame > 0)) {

        //std::random_shuffle(rand_perm_ids.begin()+context_size_, rand_perm_ids.end());
        //std::random_shuffle(rand_perm_ids.begin(), rand_perm_ids.begin() + begin_frame);
        for (int nid = begin_frame-1; (nid >= 0)
                         && (added_negatives < max_same_video_negs_); --nid) {
//          if ((rand_perm_ids[nid] < rand_perm_ids[context_size_-2])) {
//          if ((rand_perm_ids[nid] < begin_frame)) {

            for (int feature_id = 0; feature_id < (this->datum_height_-1); ++feature_id) {
              top_data[(item_id*this->datum_channels_ + this->context_size_ + added_negatives)
                        * this->datum_height_ + feature_id] =
                video_shots.shot_words(nid).float_data(feature_id);

            }
            ned_id_string += stringprintf(":%d", nid);
            added_negatives++;
 //         }
        }
      }

      /*LOG(INFO) << "Sampled negs: N" << ned_id_string << "-- C" << context_id_string
        << " -- P " << (begin_frame + (context_size_-1)*(sample_length+1))
        << " added-negs: " << added_negatives << " , sample length: " << sample_length
        << " frame-size: " << video_shots.shot_words_size();
      */

      break;

    // -------------------- Context right before the shot -----------------
    case VideoSampledShotsDataParameter_CONTEXT_PAST_CONTINUOUS_FIXED:

      if (rand_perm_ids.size() < context_size_) {
        return;
      }
      CHECK_GE(context_size_, 2);

      // Randomly pick a point in the set of frames
      //random_unique(rand_perm_ids.begin(), rand_perm_ids.begin() + (rand_perm_ids.size() - context_size + 1), 1);
      
      max_sample_length = (rand_perm_ids.size() - context_size_)/(context_size_-1);
      sample_length = (max_sample_length>=1)?(max_sample_length-1):0;

      begin_frame = rand_perm_ids.size() -
          (context_size_-1)*(sample_length) - context_size_;

     
      context_id = 0;
      for (int i = 0; i < context_size_; ++i) {
        frame_id = begin_frame + i*(sample_length + 1);
        if (i ==  (context_size_-1)) {
          for (int feature_id = 0; feature_id < this->datum_height_; ++feature_id) {
            top_data[item_id*this->datum_channels_*this->datum_height_ + feature_id] =
                     video_shots.shot_words(frame_id).float_data(feature_id);
          }
        } else {
          for (int feature_id = 0; feature_id < this->datum_height_; ++feature_id) {
            top_data[(item_id*this->datum_channels_ + context_id + 1)*this->datum_height_ + feature_id] =
                     video_shots.shot_words(frame_id).float_data(feature_id);
          }
          context_id_string += stringprintf(":%d", frame_id);
          context_id++;
        }
      }

      CHECK_EQ(context_id, context_size_-1);

      // Set the video-id
      video_id = video_shots.video_id();
      
      video_added = true;
      /*LOG(INFO) << "Top target --------------------------------> "
        << ":" << top_data[item_id*this->datum_channels_*this->datum_height_ + 1]
        << ":" << top_data[item_id*this->datum_channels_*this->datum_height_ + 10]
        << ":" << top_data[item_id*this->datum_channels_*this->datum_height_+ 100]
        << ":" << top_data[item_id*this->datum_channels_*this->datum_height_ + 500]
        << ":" << top_data[item_id*this->datum_channels_*this->datum_height_ + 2000];
      */



      // Add same video-negatives
      //LOG(INFO) << "NEGINFO: " << context_size_ << ":" << video_shots.shot_words_size() << ":" << added_negatives << ":" << max_same_video_negs_;
      added_negatives = 0;
      if ((num_negative_samples_ > 0) &&
          (begin_frame > 0)) {

        //std::random_shuffle(rand_perm_ids.begin()+context_size_, rand_perm_ids.end());
        //std::random_shuffle(rand_perm_ids.begin(), rand_perm_ids.begin() + begin_frame);
        for (int nid = begin_frame-1; (nid >= 0)
                         && (added_negatives < max_same_video_negs_); --nid) {
//          if ((rand_perm_ids[nid] < rand_perm_ids[context_size_-2])) {
//          if ((rand_perm_ids[nid] < begin_frame)) {

            for (int feature_id = 0; feature_id < (this->datum_height_-1); ++feature_id) {
              top_data[(item_id*this->datum_channels_ + this->context_size_ + added_negatives)
                        * this->datum_height_ + feature_id] =
                video_shots.shot_words(nid).float_data(feature_id);

            }
            ned_id_string += stringprintf(":%d", nid);
            added_negatives++;
 //         }
        }
      }

      /*LOG(INFO) << "Sampled negs: N" << ned_id_string << "-- C" << context_id_string
        << " -- P " << (begin_frame + (context_size_-1)*(sample_length+1))
        << " added-negs: " << added_negatives << " , sample length: " << sample_length
        << " frame-size: " << video_shots.shot_words_size();*/


      break;



    default:
      LOG(FATAL) << "Unknown context type";
  }

}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void VideoSampledShotsDataLayer<Dtype>::InternalThreadEntry() {
  VideoShots video_shots;
  CHECK(this->prefetch_data_.count());

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }

  int item_id = 0;

  //LOG(INFO) << "Starting to insert old video: ";


  while (item_id < batch_size_) {
    
    VideoShots video_shots;
    // get a blob
    switch (this->layer_param_.video_sampled_shots_data_param().backend()) {
      case VideoSampledShotsDataParameter_DB_LEVELDB:
        CHECK(iter_);
        CHECK(iter_->Valid());
        video_shots.ParseFromString(iter_->value().ToString());
        break;
      case VideoSampledShotsDataParameter_DB_LMDB:
        // Perhaps this is unecessay, we could directly take mdb_value_
        CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
        video_shots.ParseFromArray(mdb_value_.mv_data,
            mdb_value_.mv_size);
        break;
      default:
        LOG(FATAL) << "Unknown database backend";
    }

    // Check to ensure that the video shots is valid
    CHECK(video_shots.has_video_id()) << "No video id found for video shot";
    CHECK_GE(video_shots.shot_words_size(), 1) << "No shot word found: "
                                              << video_shots.video_id();

    CHECK_EQ(video_shots.shot_words(0).float_data_size(), this->datum_height_)
      << "Wrong feature size: " << video_shots.video_id();

    int num_added_negs = 0;
    bool video_added = false;
    int video_id = -1;

    // Main insertion into top-data
    //LOG(INFO) << "Trying to insert (" << item_id << ") video-id:" << video_shots.video_id();
    AddSamplesToTop(video_shots, top_data,
                    item_id, num_added_negs,
                    video_id, video_added);
    
    
    // go to the next iter
    switch (this->layer_param_.video_sampled_shots_data_param().backend()) {
      case VideoSampledShotsDataParameter_DB_LEVELDB:
        iter_->Next();
        if (!iter_->Valid()) {
          // We have reached the end. Restart from the first.
          LOG(INFO) << "Restarting data prefetching from start.";
          iter_->SeekToFirst();
        }
        break;
      case VideoSampledShotsDataParameter_DB_LMDB:
        if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
                &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
          // We have reached the end. Restart from the first.
          LOG(INFO) << "Restarting data prefetching from start.";
          CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                  &mdb_value_, MDB_FIRST), MDB_SUCCESS);
        }
        break;
      default:
        LOG(FATAL) << "Unknown database backend";
    }

    if (!video_added) {
      continue;
    }

    if (num_negative_samples_ > 0) {
      const Dtype* negatives_data = negatives_.cpu_data();
      // Sample the remaining from negatives
      RandomShuffleTopids(num_negative_samples_-num_added_negs);
      for (int negative_id = (this->context_size_ + num_added_negs);
          negative_id < (this->context_size_ + num_negative_samples_); ++negative_id) {
        int neg_id = static_cast<int>(
            this->buffer_ids_[negative_id - context_size_ - num_added_negs]);
        for (int feature_id = 0; feature_id < feature_size_; ++feature_id) {

          top_data[(item_id*this->datum_channels_ + negative_id) * this->datum_height_ + feature_id] =
           negatives_data[neg_id*feature_size_ + feature_id];
        }

        //LOG(INFO) << "neg-id: " << video_shots.video_id() << "==>" << negative_id_to_key_[neg_id];
        /*LOG(INFO) << "neg-id: " << neg_id << ":" << negatives_data[neg_id*feature_size_ + 1]
                                          << ":" << negatives_data[neg_id*feature_size_ + 10]
                                          << ":" << negatives_data[neg_id*feature_size_ + 100]
                                          << ":" << negatives_data[neg_id*feature_size_ + 500]
                                          << ":" << negatives_data[neg_id*feature_size_ + 2000];
        */
        
      }
    }

    // Add top-label
    if (this->output_labels_) {
      top_label[item_id] = video_id;
    }

    //LOG(INFO) << "Done ...";

    item_id++;

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
}

INSTANTIATE_CLASS(VideoSampledShotsDataLayer);

}  // namespace caffe
