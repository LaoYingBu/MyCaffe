#include <vector>

#include "caffe/layers/lifted_structured_loss_2_layer.hpp"
#include "caffe/layer.hpp"


namespace caffe{
	template<typename Dtype>
	void LiftedStructuredLoss2Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		Forward_cpu(bottom, top);
	}

	template<typename Dtype>
	void LiftedStructuredLoss2Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		Backward_cpu(top, propagate_down, bottom);
	}
	INSTANTIATE_LAYER_GPU_FUNCS(LiftedStructuredLoss2Layer);

} //namespace caffe