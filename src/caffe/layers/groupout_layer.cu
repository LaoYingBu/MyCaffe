#include <vector>

#include "caffe/layers/groupout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void GroupoutForward(const int n, const int group_dim, const Dtype* in,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * (mask[index / group_dim] > threshold) * scale;
  }
}

template <typename Dtype>
void GroupoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
	const int group_num = bottom[0]->count(0, 2);
	const int group_dim = bottom[0]->count(2);
	const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    unsigned int* mask =
        static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
		caffe_gpu_rng_uniform(group_num, mask);
    // set thresholds
    // NOLINT_NEXT_LINE(whitespace/operators)
    GroupoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, group_dim, bottom_data, mask, uint_thres_, scale_, top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
__global__ void GroupoutBackward(const int n, const int group_dim, const Dtype* in_diff,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * scale * (mask[index / group_dim] > threshold);
  }
}

template <typename Dtype>
void GroupoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask =
          static_cast<const unsigned int*>(rand_vec_.gpu_data());
			const int group_num = bottom[0]->count(0, 2);
			const int group_dim = bottom[0]->count(2);
      const int count = bottom[0]->count();
      // NOLINT_NEXT_LINE(whitespace/operators)
      GroupoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
          count, group_dim, top_diff, mask, uint_thres_, scale_, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GroupoutLayer);

}  // namespace caffe
