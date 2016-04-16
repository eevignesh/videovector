// This program converts a set of images to a leveldb by storing them as Datum
// proto buffers.
// Usage:
//   convert_imageset [-g] ROOTFOLDER/ LISTFILE DB_NAME RANDOM_SHUFFLE[0 or 1]
//                     [resize_height] [resize_width]
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....
// if RANDOM_SHUFFLE is 1, a random shuffle will be carried out before we
// process the file lines.
// Optional flag -g indicates the images should be read as
// single-channel grayscale. If omitted, grayscale images will be
// converted to color.

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::string;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb", "The backend for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");

struct ImageData {
  int label;
  string range_filename;
  int range_start_idx; //each line in range file corresponds to a different image (e.g. flow_x, flow_y)
};

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  // Test whether argv[1] == "-g"
  bool is_color = !FLAGS_gray;
  std::ifstream infile(argv[2]);
  std::vector<std::pair<string, ImageData> > lines;
  string filename;
  int label;
  string range_filename;
  int range_start_idx;
  while (infile >> filename >> label >> range_filename >> range_start_idx) {
    //LOG(INFO) << filename << " " << label << " " << range_filename << " " << range_start_idx;
    ImageData image_data;
    image_data.label = label;
    image_data.range_filename = range_filename;
    image_data.range_start_idx = range_start_idx;
    lines.push_back(std::make_pair(filename, image_data));
  }
  // if (FLAGS_shuffle) {
  //   // randomly shuffle data
  //   LOG(INFO) << "Shuffling data";
  //   shuffle(lines.begin(), lines.end());
  // }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  const string& db_backend = FLAGS_backend;
  const char* db_path = argv[3];

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Open new db
  // lmdb
  MDB_env *mdb_env;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn;
  // leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  leveldb::WriteBatch* batch = NULL;

  // Open db
  if (db_backend == "leveldb") {  // leveldb
    LOG(INFO) << "Opening leveldb " << db_path;
    leveldb::Status status = leveldb::DB::Open(
        options, db_path, &db);
    CHECK(status.ok()) << "Failed to open leveldb " << db_path;
    batch = new leveldb::WriteBatch();
  } else if (db_backend == "lmdb") {  // lmdb
    LOG(INFO) << "Opening lmdb " << db_path;
    CHECK_EQ(mkdir(db_path, 0744), 0)
        << "mkdir " << db_path << "failed";
    CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
        << "mdb_env_set_mapsize failed";
    CHECK_EQ(mdb_env_open(mdb_env, db_path, 0, 0664), MDB_SUCCESS)
        << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
        << "mdb_open failed";
  } else {
    LOG(FATAL) << "Unknown db backend " << db_backend;
  }

  // Storing to db
  string root_folder(argv[1]);
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int data_size;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    Datum datum;
    ImageData image_data = lines[line_id].second;
    if (!ReadImageToDatum(root_folder + lines[line_id].first,
        image_data.label, resize_height, resize_width, is_color, &datum)) {
      continue;
    }
    if (!data_size_initialized) {
      data_size = datum.channels() * datum.height() * datum.width();
      data_size_initialized = true;
    } else {
      const string& data = datum.data();
      CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
          << data.size();
    }

    string full_range_filename = root_folder + image_data.range_filename;
    std::ifstream infile(full_range_filename.c_str());
    string img_mean_str, img_min_str, img_max_str;
    float img_mean, img_min, img_max;

    int range_idx = 0;
    while (range_idx < image_data.range_start_idx) {
      infile >> img_mean_str >> img_min_str >> img_max_str;
      range_idx = range_idx + 1;
    }

    for (int channel_idx = 0; channel_idx < datum.channels(); channel_idx++) {
      infile >> img_mean_str >> img_min_str >> img_max_str;
      //LOG(INFO) << "raw " << img_mean_str << " " << img_mean << " " << img_max_str;
      sscanf(img_mean_str.c_str(), "%a", &img_mean);
      sscanf(img_min_str.c_str(), "%a", &img_min);
      sscanf(img_max_str.c_str(), "%a", &img_max);
      //fprintf(stdout, "reading %.16f %.16f %.16f\n", img_mean, img_min, img_max);
      datum.add_mean(img_mean);
      datum.add_min(img_min);
      datum.add_max(img_max);
    }

    // debug logging
    // LOG(INFO) << "---processing " << lines[line_id].first;
    // LOG(INFO) << image_data.range_filename.c_str() << " " << image_data.range_start_idx;
    // int mean_size = datum.mean_size();
    // LOG(INFO) << "datum mean size: " << mean_size;
    // for (int i = 0; i < mean_size; i++) {
    //   fprintf(stdout, "datum output %.16f %.16f %.16f\n", datum.mean(i), datum.min(i), datum.max(i));
    //   LOG(INFO) << "datum mean value: " << datum.mean(i);
    //   LOG(INFO) << "datum min value: " << datum.min(i);
    //   LOG(INFO) << "datum max value: " << datum.max(i);
    // }

    // sequential
    snprintf(key_cstr, kMaxKeyLength, "%s", 
        lines[line_id].first.c_str());
    string value;
    datum.SerializeToString(&value);
    string keystr(key_cstr);

    //LOG(INFO) << keystr;

    // Put in db
    if (db_backend == "leveldb") {  // leveldb
      batch->Put(keystr, value);
    } else if (db_backend == "lmdb") {  // lmdb
      mdb_data.mv_size = value.size();
      mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
      mdb_key.mv_size = keystr.size();
      mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
      CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
          << "mdb_put failed";
    } else {
      LOG(FATAL) << "Unknown db backend " << db_backend;
    }

    if (++count % 1000 == 0) {
      // Commit txn
      if (db_backend == "leveldb") {  // leveldb
        db->Write(leveldb::WriteOptions(), batch);
        delete batch;
        batch = new leveldb::WriteBatch();
      } else if (db_backend == "lmdb") {  // lmdb
        CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
            << "mdb_txn_commit failed";
        CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
            << "mdb_txn_begin failed";
      } else {
        LOG(FATAL) << "Unknown db backend " << db_backend;
      }
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    if (db_backend == "leveldb") {  // leveldb
      db->Write(leveldb::WriteOptions(), batch);
      delete batch;
      //delete db;
    } else if (db_backend == "lmdb") {  // lmdb
      CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
      //mdb_close(mdb_env, mdb_dbi);
      //mdb_env_close(mdb_env);
    } else {
      LOG(FATAL) << "Unknown db backend " << db_backend;
    }
    LOG(ERROR) << "Processed " << count << " files.";
  }

  // close the dbs (should be separate from above)
  if (db_backend == "leveldb") {  // leveldb
    delete db;
  } else if (db_backend == "lmdb") {  // lmdb
    mdb_close(mdb_env, mdb_dbi);
    mdb_env_close(mdb_env);
  } else {
    LOG(FATAL) << "Unknown db backend " << db_backend;
  }
  return 0;
}
