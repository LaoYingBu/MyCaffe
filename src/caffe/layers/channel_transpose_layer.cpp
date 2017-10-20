#include <vector>

#include "caffe/layers/channel_transpose_layer.hpp"

namespace caffe {

template <typename Dtype>
void ChannelTransposeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
      "allow in-place computation.";
	CHECK_GT(bottom[0]->num_axes(), 1) << "Channel transpose axis must be > 1 ";
}

template <typename Dtype>
void ChannelTransposeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const int num = bottom[0]->num();
	const int channels = bottom[0]->channels();
	const int dim = bottom[0]->count(1) / channels;
	vector<int> top_shape(1, num);
	top_shape.push_back(dim);
	top_shape.push_back(channels);
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void ChannelTransposeLayer<Dtype>::Forward_cpu(
	  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const int num = bottom[0]->num();
	const int channels = bottom[0]->channels();
	const int dim = bottom[0]->count(1) / channels;
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	for (int n = 0; n < num; n++)
	{
		for (int c = 0; c < channels; c++)
		{
			for (int d = 0; d < dim; d++)
			{			
				top_data[(n * dim + d) * channels + c] = bottom_data[(n * channels + c) * dim + d];
			}
		}
	}
}

template <typename Dtype>
void ChannelTransposeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0]) {
		const int num = bottom[0]->num();
		const int channels = bottom[0]->channels();
		const int dim = bottom[0]->count(1) / channels;
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff(); 

		for (int n = 0; n < num; n++)
		{
			for (int c = 0; c < channels; c++)
			{
				for (int d = 0; d < dim; d++)
				{
					bottom_diff[(n * channels + c) * dim + d] = top_diff[(n * dim + d) * channels + c];
				}
			}
		}
	}
}

#if CPU_ONLY
	STUB_GPU(ChannelTransposeLayer);
#endif // CPU_ONLY

INSTANTIATE_CLASS(ChannelTransposeLayer);
REGISTER_LAYER_CLASS(ChannelTranspose);

}  // namespace caffe
