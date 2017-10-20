#include <vector>

#include "caffe/layers/contrastive_data_layer.hpp"
#include "caffe/layer.hpp"


namespace caffe{
	template<typename Dtype>
	void ContrastiveDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		Forward_cpu(bottom, top);
	}

	INSTANTIATE_LAYER_GPU_FORWARD(ContrastiveDataLayer);

} //namespace caffe