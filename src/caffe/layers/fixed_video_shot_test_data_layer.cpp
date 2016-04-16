#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void FixedVideoShotTestDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     vector<Blob<Dtype>*>* top) {

  batch_size_ = this->layer_param_.fixed_video_shot_test_data_param().batch_size();
  this->datum_channels_ = this->layer_param_.fixed_video_shot_test_data_param().channels();
  this->datum_height_ = this->layer_param_.fixed_video_shot_test_data_param().height();
  this->datum_width_ = this->layer_param_.fixed_video_shot_test_data_param().width();
  this->datum_size_ = this->datum_channels_ * this->datum_height_ *
      this->datum_width_;

  CHECK_GT(batch_size_ * this->datum_size_, 0) <<
      "batch_size, channels, height, and width must be specified and"
      " positive in memory_data_param";
  (*top)[0]->Reshape(batch_size_, this->datum_channels_, this->datum_height_,
                     this->datum_width_);
  (*top)[1]->Reshape(batch_size_, 1, 1, 1);

  added_data_.Reshape(batch_size_, this->datum_channels_, this->datum_height_,
                      this->datum_width_);
  added_labels_.Reshape(batch_size_, 1, 1, 1);
  
  data_ = added_data_.mutable_cpu_data();
  labels_ =added_labels_.mutable_cpu_data();
  
  // Open the lmdb and read the features one by one
  CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
  CHECK_EQ(mdb_env_open(mdb_env_,
           this->layer_param_.fixed_video_shot_test_data_param().source().c_str(),
           MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
      << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
      << "mdb_open failed";
  CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
      << "mdb_cursor_open failed";
  LOG(INFO) << "Opening lmdb "
    << this->layer_param_.fixed_video_shot_test_data_param().source();
  
   
  // Insert the first point
  int ref_ctr = 0;
  video_shot_sentences::TestVideoShotWindows video_shots;


  CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
      MDB_SUCCESS) << "mdb_cursor_get failed";
  video_shots.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
  for (int i = 0; i < video_shots.positive_shot_words_size(); ++i) {
    CHECK_EQ(this->datum_size_,
        video_shots.positive_shot_words(i).float_data_size());
    for (int fid = 0; fid < this->datum_size_; ++fid) {
      *(data_ + added_data_.offset(ref_ctr, fid, 0, 0)) =
        video_shots.positive_shot_words(i).float_data(fid);
    }
    *(labels_ + added_labels_.offset(ref_ctr, 0, 0, 0)) = video_shots.video_id();
    //LOG(ERROR) << video_shots.video_id() << ":" << i;

    ref_ctr++;
    CHECK_LE(ref_ctr, batch_size_);
  }

  for (int i = 0; i < video_shots.negative_shot_words_size(); ++i) {
    CHECK_EQ(this->datum_size_,
        video_shots.negative_shot_words(i).float_data_size());
    for (int fid = 0; fid < this->datum_size_; ++fid) {
      *(data_ + added_data_.offset(ref_ctr, fid, 0, 0)) =
        video_shots.negative_shot_words(i).float_data(fid);
    }

    *(labels_ + added_labels_.offset(ref_ctr, 0, 0, 0)) = -1;
    //LOG(ERROR) << video_shots.video_id() << ":" << i;
    ref_ctr++;
    CHECK_LE(ref_ctr, batch_size_);

  }
 
  while (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
                == MDB_SUCCESS) {
    video_shots.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    for (int i = 0; i < video_shots.positive_shot_words_size(); ++i) {
      CHECK_EQ(this->datum_size_,
          video_shots.positive_shot_words(i).float_data_size());
      for (int fid = 0; fid < this->datum_size_; ++fid) {
        *(data_ + added_data_.offset(ref_ctr, fid, 0, 0)) =
          video_shots.positive_shot_words(i).float_data(fid);
      }
      
      *(labels_ + added_labels_.offset(ref_ctr, 0, 0, 0)) = video_shots.video_id();
      ref_ctr++;
      //LOG(ERROR) << video_shots.video_id() << ":" << i;
      CHECK_LE(ref_ctr, batch_size_);
    }

    for (int i = 0; i < video_shots.negative_shot_words_size(); ++i) {
      CHECK_EQ(this->datum_size_,
          video_shots.negative_shot_words(i).float_data_size());
      for (int fid = 0; fid < this->datum_size_; ++fid) {
        *(data_ + added_data_.offset(ref_ctr, fid, 0, 0)) =
          video_shots.negative_shot_words(i).float_data(fid);
      }

      *(labels_ + added_labels_.offset(ref_ctr, 0, 0, 0)) = -1;
      //LOG(ERROR) << video_shots.video_id() << ":" << i;
      ref_ctr++;
      CHECK_LE(ref_ctr, batch_size_);
    }

  }

  LOG(INFO) << "--- Found " << ref_ctr << " reference points " << " out of "
    << batch_size_ << " stated ref points.";
  
  CHECK_EQ(ref_ctr, batch_size_);

  // Close the lmdb
  mdb_cursor_close(mdb_cursor_);
  mdb_close(mdb_env_, mdb_dbi_);
  mdb_txn_abort(mdb_txn_);
  mdb_env_close(mdb_env_);


}

template <typename Dtype>
void FixedVideoShotTestDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  
  (*top)[0]->set_cpu_data(added_data_.mutable_cpu_data());
  (*top)[1]->set_cpu_data(added_labels_.mutable_cpu_data());
}

INSTANTIATE_CLASS(FixedVideoShotTestDataLayer);

}  // namespace caffe
