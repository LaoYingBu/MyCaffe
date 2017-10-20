#include <vector>

#include "caffe/layers/norm_by_channel_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NormByChannelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int spatial_dim = bottom[0]->height() * bottom[0]->width();

	caffe_gpu_powx(bottom[0]->count(), bottom_data, Dtype(2),
		length_.mutable_gpu_data());  // X^2
	for (int item = 0; item < num; item++)
	{
		caffe_gpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, 1,
			length_.gpu_data() + item * bottom[0]->count(1), sum_multiplier_.gpu_data(),
			0., sumvec_.mutable_gpu_data() + item * sumvec_.count(1));  // Sum(X^2)
	}
	caffe_gpu_powx(sumvec_.count(), sumvec_.gpu_data(), Dtype(0.5),
		sumvec_.mutable_gpu_data());
	caffe_gpu_add_scalar(sumvec_.count(), eps_, sumvec_.mutable_gpu_data());
	for (int item = 0; item < num; item++)
	{
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim, 1, 1.,
			sum_multiplier_.gpu_data(), sumvec_.gpu_data() + item * sumvec_.count(1), 0.,
			length_.mutable_gpu_data() + item * bottom[0]->count(1));
	}
	caffe_gpu_div(bottom[0]->count(), bottom_data, length_.gpu_data(), top_data);
}

template <typename Dtype>
void NormByChannelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int spatial_dim = bottom[0]->height() * bottom[0]->width();

	//caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

	caffe_gpu_mul(bottom[0]->count(), top_data, top_diff, bottom_diff);
	for (int item = 0; item < num; item++)
	{
		caffe_gpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, 1,
			bottom_diff + item * bottom[0]->count(1), sum_multiplier_.gpu_data(),
			0., sumvec_.mutable_gpu_data() + item * sumvec_.count(1));
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim, 1, 1.,
			sum_multiplier_.gpu_data(), sumvec_.gpu_data() + item * sumvec_.count(1), 0.,
			bottom_diff + item * bottom[0]->count(1));
	}
	caffe_gpu_mul(bottom[0]->count(), top_data, bottom_diff, bottom_diff);
	caffe_gpu_sub(bottom[0]->count(), top_diff, bottom_diff, bottom_diff);
	caffe_gpu_div(bottom[0]->count(), bottom_diff, length_.gpu_data(), bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(NormByChannelLayer);

}  // namespace caffe
