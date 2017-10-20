// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/groupout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GroupoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
	CHECK_EQ(bottom[0]->num_axes(), 4) << "Groupout only accept map input.";
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Dtype>
void GroupoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
	vector<int> group_shape(1, bottom[0]->count(0,2));
	rand_vec_.Reshape(group_shape);
}

template <typename Dtype>
void GroupoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  unsigned int* mask = rand_vec_.mutable_cpu_data();
	const int group_num = bottom[0]->count(0, 2);
	const int group_dim = bottom[0]->count(2);
  if (this->phase_ == TRAIN) {
    // Create random numbers
		caffe_rng_bernoulli(group_num, 1. - threshold_, mask);
		for (int i = 0; i < group_num; ++i)
			for (int j = 0; j < group_dim; ++j)
			{
				top_data[i * group_dim + j] = bottom_data[i * group_dim + j] * mask[i] * scale_;
			}
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void GroupoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask = rand_vec_.cpu_data();
			const int group_num = bottom[0]->count(0, 2);
			const int group_dim = bottom[0]->count(2);
			for (int i = 0; i < group_num; ++i)
				for (int j = 0; j < group_dim; ++j)
				{
					bottom_diff[i * group_dim + j] = top_diff[i * group_dim + j] * mask[i] * scale_;
				}
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(GroupoutLayer);
#endif

INSTANTIATE_CLASS(GroupoutLayer);
REGISTER_LAYER_CLASS(Groupout);

}  // namespace caffe
