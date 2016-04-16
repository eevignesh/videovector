#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "hdf5.h"
#include "leveldb/db.h"
#include "lmdb.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/proto/video_shot_sentences.pb.h"
#include "caffe/proto/tracking_windows.pb.h"

namespace caffe {

#define HDF5_DATA_DATASET_NAME "data"
#define HDF5_DATA_LABEL_NAME "label"


/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
  explicit BaseDataLayer(const LayerParameter& param);
  virtual ~BaseDataLayer() {}
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {}
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {}

  int datum_channels() const { return datum_channels_; }
  int datum_height() const { return datum_height_; }
  int datum_width() const { return datum_width_; }
  int datum_size() const { return datum_size_; }

 protected:
  TransformationParameter transform_param_;
  DataTransformer<Dtype> data_transformer_;
  int datum_channels_;
  int datum_height_;
  int datum_width_;
  int datum_size_;
  Blob<Dtype> data_mean_;
  const Dtype* mean_;
  Caffe::Phase phase_;
  bool output_labels_;
};

template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingDataLayer(const LayerParameter& param)
      : BaseDataLayer<Dtype>(param) {}
  virtual ~BasePrefetchingDataLayer() {}
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  // The thread's function
  virtual void InternalThreadEntry() {}

 protected:
  Blob<Dtype> prefetch_data_;
  Blob<Dtype> prefetch_label_;
};

template <typename Dtype>
class DataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit DataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~DataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void InternalThreadEntry();

  // LEVELDB
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
};


/**
 * @brief Provides data to the Net from memory.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class FixedVideoShotTestDataLayer : public BaseDataLayer<Dtype> {
 public:
  explicit FixedVideoShotTestDataLayer(const LayerParameter& param)
      : BaseDataLayer<Dtype>(param) {}
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_FIXED_VIDEO_SHOT_TEST_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

  int batch_size() { return batch_size_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  int batch_size_;
  Dtype* data_;
  Dtype* labels_;
  Blob<Dtype> added_data_;
  Blob<Dtype> added_labels_;

  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;

};


/**
 * @brief Reads and provides video shot windows to the net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */

template <typename Dtype>
class VideoShotWindowTestDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit VideoShotWindowTestDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~VideoShotWindowTestDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_VIDEO_SHOT_WINDOW_TEST_DATA;
  }

  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void InternalThreadEntry();

  int feature_size_;
  int context_size_;
  int positive_size_;
  int negative_size_;

  // LEVELDB
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;

};

/**
 * @brief Reads and provides video shots to the net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */

template <typename Dtype>
class VideoSampledShotsDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit VideoSampledShotsDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~VideoSampledShotsDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_VIDEO_SAMPLED_SHOTS_DATA;
  }

  virtual inline int AddToBuffer(const Datum data);
  virtual inline void RandomShuffleTopids(int n);
  virtual inline void AddSamplesToTop(const video_shot_sentences::VideoShots& video_shots,
      Dtype* top_data, const int &item_id, int& num_added_negs, int& video_id, bool& video_added);
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void InternalThreadEntry();


  int feature_size_;
  int context_size_;
  int batch_size_;

  bool output_shot_distance_;

  // LEVELDB
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;

  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;

  // For negative smapling
  Blob<Dtype> negatives_;
  std::set<string> negative_keys_set_; // faster lookup
  std::vector<string> negative_id_to_key_;
  vector<Dtype> buffer_ids_;
  int max_buffer_size_;
  size_t negative_swap_percentage_; // should be less than 100
  int num_negative_samples_;
  vector <int> neg_added_from_same_video_;
  int max_same_video_negs_;

  // LEVELDB-NEGATIVES
  shared_ptr<leveldb::DB> db_neg_;
  shared_ptr<leveldb::Iterator> iter_neg_;

  // LMDB-NEGATIVES
  MDB_env* mdb_env_neg_;
  MDB_dbi mdb_dbi_neg_;
  MDB_txn* mdb_txn_neg_;
  MDB_cursor* mdb_cursor_neg_;
  MDB_val mdb_key_neg_, mdb_value_neg_;

};


/**
 * @brief Reads and provides video shot windows to the net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */

template <typename Dtype>
class TrackingWindowsSocialDataLayer : public Layer<Dtype>, public InternalThread {
 public:
  explicit TrackingWindowsSocialDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {};
//      : BaseDataLayer<Dtype>(param) {}
  virtual ~TrackingWindowsSocialDataLayer();

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_TRACKING_WINDOWS_SOCIAL_DATA;
  }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top); 
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {}

  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 5; }
  virtual inline int MaxTopBlobs() const { return 7; }

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();

 protected:

  // The thread's function
  virtual void InternalThreadEntry();

  bool output_labels_;
  bool output_scene_ids_;

  Caffe::Phase phase_;

  Blob<Dtype> prefetch_data_0_;
  Blob<Dtype> prefetch_data_1_;
  Blob<Dtype> prefetch_data_2_;
  Blob<Dtype> prefetch_data_3_;
  Blob<Dtype> prefetch_data_4_;

  Blob<Dtype> prefetch_label_;
  Blob<Dtype> prefetch_scene_;

  int feature_size_observed_;
  int feature_size_predicted_;
  int batch_size_;
  int temporal_observed_size_;
  int temporal_predicted_size_;
  int use_static_scene_;
  int max_number_positions_;
  int num_positions_;
  int prev_track_id_;
  tracking_windows::TrackingWindow prev_tracking_window_;

  // LEVELDB
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;

};



/**
 * @brief Reads and provides video shot windows to the net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */

template <typename Dtype>
class TrackingWindowsDataLayer : public Layer<Dtype>, public InternalThread {
 public:
  explicit TrackingWindowsDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {};
//      : BaseDataLayer<Dtype>(param) {}
  virtual ~TrackingWindowsDataLayer();

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_TRACKING_WINDOWS_DATA;
  }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top); 
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {}

  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 4; }
  virtual inline int MaxTopBlobs() const { return 6; }

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();

 protected:

  // The thread's function
  virtual void InternalThreadEntry();

  bool output_labels_;
  bool output_scene_ids_;

  Caffe::Phase phase_;

  Blob<Dtype> prefetch_data_0_;
  Blob<Dtype> prefetch_data_1_;
  Blob<Dtype> prefetch_data_2_;
  Blob<Dtype> prefetch_data_3_;
  Blob<Dtype> prefetch_label_;
  Blob<Dtype> prefetch_scene_;

  int feature_size_observed_;
  int feature_size_predicted_;
  int batch_size_;
  int temporal_observed_size_;
  int temporal_predicted_size_;
  int use_static_scene_;
  int max_number_positions_;
  int num_positions_;
  int prev_track_id_;
  tracking_windows::TrackingWindow prev_tracking_window_;

  // LEVELDB
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;

};


/**
 * @brief Reads and provides video shots to the net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */

template <typename Dtype>
class VideoShotsDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit VideoShotsDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~VideoShotsDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_VIDEO_SHOTS_DATA;
  }

  virtual inline int AddToBuffer(const Datum data);
  virtual inline void RandomShuffleTopids(int n);
  virtual inline void InsertIntoQueue(const video_shot_sentences::VideoShots& video_shots,
      Dtype* top_data, int &item_id);
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void InternalThreadEntry();


  int feature_size_;
  int context_size_;
  int batch_size_;

  bool output_shot_distance_;

  // LEVELDB
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;

  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;

  // For negative smapling
  Blob<Dtype> negatives_;
  std::set<string> negative_keys_set_; // faster lookup
  std::vector<string> negative_id_to_key_;
  vector<Dtype> buffer_ids_;
  int max_buffer_size_;
  size_t negative_swap_percentage_; // should be less than 100
  int num_negative_samples_;
  vector <int> neg_added_from_same_video_;
  int max_same_video_negs_;

  // LEVELDB-NEGATIVES
  shared_ptr<leveldb::DB> db_neg_;
  shared_ptr<leveldb::Iterator> iter_neg_;

  // LMDB-NEGATIVES
  MDB_env* mdb_env_neg_;
  MDB_dbi mdb_dbi_neg_;
  MDB_txn* mdb_txn_neg_;
  MDB_cursor* mdb_cursor_neg_;
  MDB_val mdb_key_neg_, mdb_value_neg_;


  // State-parameter of reading
  int target_ctr_;
  int context_ctr_;
  video_shot_sentences::VideoShots current_video_shots_;
  vector<Dtype> video_ids_;
};


/**
 * @brief Reads and provides video shot windows to the net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */

template <typename Dtype>
class VideoShotWindowDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit VideoShotWindowDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~VideoShotWindowDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_VIDEO_SHOT_WINDOW_DATA;
  }

  virtual inline int AddToBuffer(const Datum data);
  virtual inline void RandomShuffleTopids(int n);

  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void InternalThreadEntry();


  int feature_size_;
  int context_size_;

  // TEXT_VIDEO_ID
  vector<int> test_video_ids_;
  int next_video_id_;

  // LEVELDB
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;

  // For negative smapling
  Blob<Dtype> negatives_;
  std::set<string> negative_keys_set_; // faster lookup
  std::vector<string> negative_id_to_key_;
  vector<Dtype> buffer_ids_;
  int max_buffer_size_;
  size_t negative_swap_percentage_; // should be less than 100
  int num_negative_samples_;

  // LEVELDB-NEGATIVES
  shared_ptr<leveldb::DB> db_neg_;
  shared_ptr<leveldb::Iterator> iter_neg_;

  // LMDB-NEGATIVES
  MDB_env* mdb_env_neg_;
  MDB_dbi mdb_dbi_neg_;
  MDB_txn* mdb_txn_neg_;
  MDB_cursor* mdb_cursor_neg_;
  MDB_val mdb_key_neg_, mdb_value_neg_;

};



/**
 * @brief Provides data to the Net generated by a Filler.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class DummyDataLayer : public Layer<Dtype> {
 public:
  explicit DummyDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {}

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_DUMMY_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {}

  vector<shared_ptr<Filler<Dtype> > > fillers_;
  vector<bool> refill_;
};

/**
 * @brief Provides data to the Net from HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class HDF5DataLayer : public Layer<Dtype> {
 public:
  explicit HDF5DataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~HDF5DataLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {}

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_HDF5_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {}
  virtual void LoadHDF5FileData(const char* filename);

  std::vector<std::string> hdf_filenames_;
  unsigned int num_files_;
  unsigned int current_file_;
  hsize_t current_row_;
  Blob<Dtype> data_blob_;
  Blob<Dtype> label_blob_;
};

/**
 * @brief Write blobs to disk as HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class HDF5OutputLayer : public Layer<Dtype> {
 public:
  explicit HDF5OutputLayer(const LayerParameter& param);
  virtual ~HDF5OutputLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {}
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {}

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_HDF5_OUTPUT;
  }
  // TODO: no limit on the number of blobs
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

  inline std::string file_name() const { return file_name_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void SaveBlobs();

  std::string file_name_;
  hid_t file_id_;
  Blob<Dtype> data_blob_;
  Blob<Dtype> label_blob_;
};

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_IMAGE_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();

  vector<std::pair<std::string, int> > lines_;
  int lines_id_;
};

/**
 * @brief Provides data to the Net from memory.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class MemoryDataLayer : public BaseDataLayer<Dtype> {
 public:
  explicit MemoryDataLayer(const LayerParameter& param)
      : BaseDataLayer<Dtype>(param), has_new_data_(false) {}
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_MEMORY_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

  virtual void AddDatumVector(const vector<Datum>& datum_vector);

  // Reset should accept const pointers, but can't, because the memory
  //  will be given to Blob, which is mutable
  void Reset(Dtype* data, Dtype* label, int n);

  int batch_size() { return batch_size_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  int batch_size_;
  Dtype* data_;
  Dtype* labels_;
  int n_;
  int pos_;
  Blob<Dtype> added_data_;
  Blob<Dtype> added_label_;
  bool has_new_data_;
};

/**
 * @brief Provides data to the Net from windows of images files, specified
 *        by a window data file.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class WindowDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit WindowDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~WindowDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_WINDOW_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual unsigned int PrefetchRand();
  virtual void InternalThreadEntry();

  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<std::pair<std::string, vector<int> > > image_database_;
  enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2, NUM };
  vector<vector<float> > fg_windows_;
  vector<vector<float> > bg_windows_;
};

/**
 * @brief Flexible Data Layer
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class FlexibleDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit FlexibleDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~FlexibleDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void InternalThreadEntry();

  // LEVELDB
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
  // Flexi LMDB
  MDB_env* flexi_mdb_env_;
  MDB_dbi flexi_mdb_dbi_;
  MDB_txn* flexi_mdb_txn_;
  MDB_cursor* flexi_mdb_cursor_;
  MDB_val flexi_mdb_key_, flexi_mdb_value_;

  int item_channels_;
};

/**
 * @brief Provides data to the Net from windows of images files, specified
 *        by a window data file.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
// template <typename Dtype>
// class TemporalDataLayer : public Layer<Dtype>, public InternalThread {
//  public:
//   explicit TemporalDataLayer(const LayerParameter& param)
//       : Layer<Dtype>(param) {}
//   virtual ~TemporalDataLayer();
//   virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
//       vector<Blob<Dtype>*>* top);

//   virtual inline LayerParameter_LayerType type() const {
//     return LayerParameter_LayerType_TEMPORAL_DATA;
//   }
//   virtual inline int ExactNumBottomBlobs() const { return 0; }
//   virtual inline int MinTopBlobs() const { return 1; }
//   virtual inline int MaxTopBlobs() const { return 2; }

//  protected:
//   virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//       vector<Blob<Dtype>*>* top);
//   virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//       vector<Blob<Dtype>*>* top);
//   virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
//       const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {}
//   virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//       const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {}

//   virtual void CreatePrefetchThread();
//   virtual void JoinPrefetchThread();
//   virtual unsigned int PrefetchRand();
//   // The thread's function
//   virtual void InternalThreadEntry();

//   virtual void ReadDatumIntoPrefetch(Datum& datum, Dtype* top_data, const int channel_start_idx, const int crop_size, const int scale, const int h_off, const int w_off, const bool do_mirror, const int item_id, const int temporal_l);

//   shared_ptr<Caffe::RNG> prefetch_rng_;

//   // LEVELDB
//   shared_ptr<leveldb::DB> db_;
//   shared_ptr<leveldb::Iterator> iter_;
//   // LMDB
//   MDB_env* mdb_env_;
//   MDB_dbi mdb_dbi_;
//   MDB_txn* mdb_txn_;
//   MDB_cursor* mdb_cursor_;
//   MDB_val mdb_key_, mdb_value_;

//   MDB_env* off_mdb_env_;
//   MDB_dbi off_mdb_dbi_;
//   MDB_txn* off_mdb_txn_;
//   MDB_cursor* off_mdb_cursor_;
//   MDB_val off_mdb_key_, off_mdb_value_;

//   MDB_env* ofb_mdb_env_;
//   MDB_dbi ofb_mdb_dbi_;
//   MDB_txn* ofb_mdb_txn_;
//   MDB_cursor* ofb_mdb_cursor_;
//   MDB_val ofb_mdb_key_, ofb_mdb_value_;

//   int datum_channels_;
//   int datum_height_;
//   int datum_width_;
//   int datum_size_;
//   Blob<Dtype> prefetch_data_;
//   Blob<Dtype> prefetch_label_;
//   Blob<Dtype> data_mean_;
//   bool output_labels_;
//   Caffe::Phase phase_;
// };

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
