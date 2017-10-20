#include <vector>

#include "caffe/layers/norm_by_channel_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NormByChannelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	top[0]->Reshape(bottom[0]->shape());
	sumvec_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
	length_.Reshape(bottom[0]->shape());
	sum_multiplier_.Reshape(1, bottom[0]->channels(), 1, 1);

  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  eps_ = this->layer_param_.mvn_param().eps();
}

template <typename Dtype>
void NormByChannelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int spatial_dim = bottom[0]->height() * bottom[0]->width();

	caffe_powx(bottom[0]->count(), bottom_data, Dtype(2),
		length_.mutable_cpu_data());  // X^2
	for (int item = 0; item < num; item++)
	{
		caffe_cpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, 1,
			length_.cpu_data() + item * bottom[0]->count(1), sum_multiplier_.cpu_data(),
			0., sumvec_.mutable_cpu_data() + item * sumvec_.count(1));  // Sum(X^2)
	}
	caffe_powx(sumvec_.count(), sumvec_.cpu_data(), Dtype(0.5),
		sumvec_.mutable_cpu_data());
	caffe_add_scalar(sumvec_.count(), eps_, sumvec_.mutable_cpu_data());
	for (int item = 0; item < num; item++)
	{
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim, 1, 1.,
			sum_multiplier_.cpu_data(), sumvec_.cpu_data() + item * sumvec_.count(1), 0.,
			length_.mutable_cpu_data() + item * bottom[0]->count(1));
	}
	caffe_div(bottom[0]->count(), bottom_data, length_.cpu_data(), top_data);
}

template <typename Dtype>
void NormByChannelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int spatial_dim = bottom[0]->height() * bottom[0]->width();

	//caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

	caffe_mul(bottom[0]->count(), top_data, top_diff, bottom_diff);
	for (int item = 0; item < num; item++)
	{
		caffe_cpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, 1,
			bottom_diff + item * bottom[0]->count(1), sum_multiplier_.cpu_data(),
			0., sumvec_.mutable_cpu_data() + item * sumvec_.count(1));
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim, 1, 1.,
			sum_multiplier_.cpu_data(), sumvec_.cpu_data() + item * sumvec_.count(1), 0.,
			bottom_diff + item * bottom[0]->count(1));
	}
	caffe_mul(bottom[0]->count(), top_data, bottom_diff, bottom_diff);
	caffe_sub(bottom[0]->count(), top_diff, bottom_diff, bottom_diff);
	caffe_div(bottom[0]->count(), bottom_diff, length_.cpu_data(), bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(NormByChannelLayer);
#endif

INSTANTIATE_CLASS(NormByChannelLayer);
REGISTER_LAYER_CLASS(NormByChannel);

}  // namespace caffe
