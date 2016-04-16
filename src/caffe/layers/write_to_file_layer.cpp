#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/vignesh_util.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void WriteToFileLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {

  output_file_ = this->layer_param_.write_to_file_param().output_file();
  feat_size_ = this->layer_param_.write_to_file_param().feat_size();
  
  feat_size_ = (feat_size_<=0)?(bottom[0]->width()*bottom[0]->height()):feat_size_;
}

template <typename Dtype>
void WriteToFileLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_GE(bottom[0]->count()/(bottom[0]->num() * bottom[0]->channels()), feat_size_)
      << "The feature size should be larger";
}


template <typename Dtype>
void WriteToFileLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  ofstream stats_output;

  if (output_file_ != "") {
    stats_output.open(output_file_);
    stats_output << "#batch_id,channel_id,features(1.." << feat_size_ << ")" << std::endl; 
  }

  //const Dtype* bottom_data = bottom[0]->cpu_data();
  
  if (output_file_ != "") {
    for (int c = 0; c < bottom[0]->channels(); ++c) {
      for (int b = 0; b < bottom[0]->num(); ++b) {
        stats_output << c << "," << b;
        for (int f = 0; f < feat_size_; ++f) {
          stats_output << ","
            << static_cast<float>(bottom[0]->data_at(b, c, f, 0));
        }
        stats_output << std::endl;
      }
    }
  }    
  if (output_file_ != "") {
    stats_output.close();
  }

}

INSTANTIATE_CLASS(WriteToFileLayer);

}  // namespace caffe
