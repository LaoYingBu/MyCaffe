#include <algorithm>
#include <vector>

#include "caffe/layers/channel_transpose_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	__global__ void ChannelTrans(const int nthreads, const int num,
		const int first_axis, const int second_axis, const Dtype* src_data, 
		Dtype* dst_data) {

		CUDA_KERNEL_LOOP(index, nthreads) {
			const int item = index / (first_axis * second_axis);
			const int j = (index / first_axis) % second_axis;
			const int k = index % first_axis;

			dst_data[index] = src_data[(item * first_axis + k) * second_axis + j];
		}
	}
	
	template <typename Dtype>
	void ChannelTransposeLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		const int num = bottom[0]->num();
		const int channels = bottom[0]->channels();
		const int dim = bottom[0]->count(1) / channels;
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		Dtype nthreads = num * channels * dim;
		
		ChannelTrans<Dtype> << <CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS >> >(
			nthreads, num, channels, dim, bottom_data, top_data);
	}

	template <typename Dtype>
	void ChannelTransposeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	
		if (propagate_down[0]) {
			const int num = bottom[0]->num();
			const int channels = bottom[0]->channels();
			const int dim = bottom[0]->count(1) / channels;
			const Dtype* top_diff = top[0]->gpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
			Dtype nthreads = num * channels * dim;

			ChannelTrans<Dtype> << <CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS >> >(
				nthreads, num, dim, channels, top_diff, bottom_diff);
		}

	}

INSTANTIATE_LAYER_GPU_FUNCS(ChannelTransposeLayer);

}  // namespace caffe
