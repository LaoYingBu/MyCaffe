#include <vector>

#include "caffe/layers/quantized_loss_layer.hpp"
#include "caffe/layer.hpp"


namespace caffe{
	template<typename Dtype>
	void QuantizedLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		Forward_cpu(bottom, top);
	}

	template<typename Dtype>
	void QuantizedLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		Backward_cpu(top, propagate_down, bottom);
	}
	INSTANTIATE_LAYER_GPU_FUNCS(QuantizedLossLayer);

} //namespace caffe