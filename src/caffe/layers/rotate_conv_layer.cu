#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/rotate_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void RotateMap_gpu(const int nthreads,
	  const Dtype* const src, Dtype* const dst, const int outer_dim, 
		const int height, const int width, const int idx0, const int hstep,
		const int wstep, bool inverse) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int w = index % width;
		const int h = (index / width) % height;
		const int d = (index / width / height) % outer_dim;
		if (inverse) 
		{
			dst[index] += src[d * height * width + h * hstep + w * wstep];
		}
		else 
		{
			dst[d * height * width + h * hstep + w * wstep] += src[index];
		}
	}
}

template <typename Dtype>
void RotateConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	int outer_dim = bottom[0]->num() * bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	int count = bottom[0]->count();
	int idx0, hstep, wstep;

	for (int orient = 0; orient < exnum_; orient++)
	{
		caffe_gpu_set(convolution_inputs_[orient]->count(), Dtype(0), convolution_inputs_[orient]->mutable_gpu_data());
		caffe_gpu_set(eltwise_inputs_[orient]->count(), Dtype(0), eltwise_inputs_[orient]->mutable_gpu_data());
	}

	for (int orient = 0; orient < exnum_; orient++)
	{
		RotateIndex(height, width, orient, idx0, hstep, wstep);
		RotateMap_gpu<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			bottom[0]->gpu_data(), convolution_inputs_[orient]->mutable_gpu_data(),
			outer_dim, height, width, idx0, hstep, wstep, false);
	}
	convolution_layer_->forward_gpu(convolution_bottom_vec_, convolution_top_vec_);

	height = top[0]->height();
	width = top[0]->width();
	count = top[0]->count();
	for (int orient = 0; orient < exnum_; orient++)
	{
		RotateIndex(height, width, orient, idx0, hstep, wstep);
		RotateMap_gpu<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			convolution_outputs_[orient]->gpu_data(), eltwise_inputs_[orient]->mutable_gpu_data(),
			outer_dim, height, width, idx0, hstep, wstep, true);
	}
	eltwise_layer_->forward_gpu(eltwise_bottom_vec_, eltwise_top_vec_);
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void RotateConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	int outer_dim = bottom[0]->num() * bottom[0]->channels();
	int height = top[0]->height();
	int width = top[0]->width();
	int count = top[0]->count();

	for (int orient = 0; orient < exnum_; orient++) {
		caffe_gpu_set(eltwise_outputs_[orient]->count(), Dtype(0), eltwise_outputs_[orient]->mutable_gpu_diff());
		caffe_gpu_set(convolution_outputs_[orient]->count(), Dtype(0), convolution_outputs_[orient]->mutable_gpu_diff());
	}
	caffe_gpu_set(bottom[0].count(), Dtype(0), bottom[0].mutable_gpu_diff());

	eltwise_layer_->backward_gpu(eltwise_top_vec_, propagate_down, eltwise_bottom_vec_);
	for (int orient = 0; orient < exnum_; orient++) {
		RotateIndex(height, width, orient, idx0, hstep, wstep);
		RotateMap_gpu<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			eltwise_inputs_[orient]->gpu_diff(), convolution_outputs_[orient]->mutable_gpu_diff(),
			outer_dim, height, width, idx0, hstep, wstep, false);
	}

	convolution_layer_->backward_gpu(convolution_top_vec_, propagate_down, convolution_bottom_vec_);
	height = bottom[0]->height();
	width = bottom[0]->width();
	count = bottom[0]->count();
	for (int orient = 0; orient < exnum_; orient++) {
		RotateIndex(height, width, orient, idx0, hstep, wstep);
		RotateMap_gpu<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			convolution_inputs_[orient].gpu_diff(), bottom[0]->mutable_gpu_diff(),
			outer_dim, height, width, idx0, hstep, wstep, true);
	}
	CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(PoolingLayer);


}  // namespace caffe
