#include <vector>

#include "caffe/layers/quantized_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void QuantizedLossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
	top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void QuantizedLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	int batch_size = bottom[0]->num();
  int count = bottom[0]->count();
  Dtype loss = 0;
	for (int i = 0; i < count; i++)
	{
		Dtype x = bottom[0]->cpu_data()[i];
		loss += (x >= 0 ? ((x - 1) * (x - 1)) : ((x + 1) * (x + 1)));
	}
  top[0]->mutable_cpu_data()[0] = loss / batch_size / 2;
}

template <typename Dtype>
void QuantizedLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	int batch_size = bottom[0]->num();
	if (propagate_down[0]) {
		int count = bottom[0]->count();
		for (int i = 0; i < count; i++)
		{
			Dtype x = bottom[0]->cpu_data()[i];
			bottom[0]->mutable_cpu_diff()[i] = (x >= 0 ? (x - 1) : (x + 1));
		}

		caffe_scal(count, top[0]->cpu_diff()[0] / batch_size, bottom[0]->mutable_cpu_diff());
	}
}

#ifdef CPU_ONLY
STUB_GPU(QuantizedLossLayer);
#endif

INSTANTIATE_CLASS(QuantizedLossLayer);
REGISTER_LAYER_CLASS(QuantizedLoss);

}  // namespace caffe
