#include <string>
#include <vector>

#include "fcntl.h"
#include "google/protobuf/text_format.h"
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/vignesh_util.hpp"


using namespace caffe;

int main(int argc, char** argv) {

  int height, width;
  float r, g, b;
  string output_file;

  if (argc != 7) {
    LOG(ERROR) << "Need 6 inputs: R-value G-value B-value im_height im_width output_file";
    LOG(ERROR) << "Instead got only: " << argc;
    return -1;
  } else {
    r = atof(argv[1]);
    g = atof(argv[2]);
    b = atof(argv[3]);
    height = atoi(argv[4]);
    width = atoi(argv[5]);
    output_file = string(argv[6]);
    LOG(ERROR) << "R: " << r
               << "G: " << g
               << "B: " << b
               << "h: " << height
               << "w: " << width
               << "ofile: " << output_file;
  }

  Blob<double> mean_blob(1, 3, height, width);
  mean_blob.Reshape(1, 3, height, width);
  auto* blob_data = mean_blob.mutable_cpu_data();

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      blob_data[mean_blob.offset(0, 0, i, j)] = r;
      blob_data[mean_blob.offset(0, 1, i, j)] = g;
      blob_data[mean_blob.offset(0, 2, i, j)] = b;   
    }
  }

  BlobProto output_blob_proto;
  mean_blob.ToProto(&output_blob_proto);
  WriteProtoToBinaryFile(output_blob_proto, output_file);

  /*BlobProto blob_proto_read;
  Blob<double> data_mean_;
  string test_mean_file = "./misc/imagenet_mean.binaryproto";
  ReadProtoFromBinaryFileOrDie(test_mean_file.c_str(), &blob_proto_read);
  data_mean_.FromProto(blob_proto_read);
  CHECK_EQ(data_mean_.num(), 1);
  CHECK_EQ(data_mean_.channels(), 3);
  CHECK_EQ(data_mean_.height(), 256);
  CHECK_EQ(data_mean_.width(), 256);


  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      LOG(ERROR) << data_mean_.data_at(0,0,i,j) << ":" << i << ":" << j;
    }
  }*/

  LOG(ERROR) << "Written to file successfully!";

  return 0;
}
