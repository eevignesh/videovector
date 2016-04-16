#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ClassificationStatsLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  num_classes_ = this->layer_param_.classification_stats_param().num_classes();
}

template <typename Dtype>
void ClassificationStatsLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(num_classes_, bottom[0]->count()/ bottom[0]->num())
      << "The input score count should be equal to number of classes.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);

  (*top)[0]->Reshape(1, num_classes_, 1, 1); // Accuracy per class 
  (*top)[1]->Reshape(1, num_classes_, 1, 1); // Mean ap for each individual class
  (*top)[2]->Reshape(1, 1, 1, 1); // Total accuracy across all classes
}

template <typename Dtype>
void ClassificationStatsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();

  vector<std::pair<Dtype, bool> > dummy_vector(num, std::make_pair(0, false));
  vector<vector<std::pair<Dtype, bool> > > score_to_class_vector(num_classes_, dummy_vector);
  vector<Dtype> mean_ap(num_classes_, 0);
  vector<Dtype> accuracy_vector(num_classes_, 0);
  vector<int> class_count(num_classes_, 0);
  Dtype total_accuracy = 0;

  for (int i = 0; i < num; ++i) {
    int max_id = 0;
    float max_val = bottom_data[i * num_classes_];
    int true_label = static_cast<int>(bottom_label[i]);
    class_count[true_label]++;

    for (int j = 0; j < num_classes_; ++j) {
      score_to_class_vector[j].push_back(
          std::make_pair(bottom_data[i * num_classes_ + j], true_label==j));
      if (bottom_data[i * num_classes_ + j] > max_val) {
        max_id  = j;
        max_val = bottom_data[i * num_classes_ + j];
      }
    }
    if (max_id == true_label) {
      ++accuracy_vector[true_label];
      ++total_accuracy;
    }
  }

  for (int i = 0; i < num_classes_; ++i) {
    if (class_count[i] > 0) {
      (*top)[0]->mutable_cpu_data()[i] = accuracy_vector[i] / class_count[i]; // Set accuracy
      std::sort(
        score_to_class_vector[i].begin(), score_to_class_vector[i].end(),
        std::greater<std::pair<Dtype, bool> >());

      // Compute the average precision
      Dtype precision = 0;
      Dtype num_correct = 0;
      for (int j = 0; j < num; ++j) {
        if (score_to_class_vector[i][j].second) {
          ++num_correct;
          precision += (num_correct/(j+1));
        }
      }
      (*top)[1]->mutable_cpu_data()[i] = precision / class_count[i];

    } else {
      (*top)[0]->mutable_cpu_data()[i] = 0; // Accuracy of the class
      (*top)[1]->mutable_cpu_data()[i] = 0; // Mean ap of the class
    }
  }

  (*top)[2]->mutable_cpu_data()[0] = total_accuracy / num;
  // This layer should not be used as a loss function.
}

INSTANTIATE_CLASS(ClassificationStatsLayer);

}  // namespace caffe
