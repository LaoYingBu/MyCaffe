#include <vector>

#include "caffe/layers/pruned_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PrunedConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	if (!pruned_setup_)
	{
		PrunedSetUp();
	}

	caffe_gpu_mul(blobs_[0]->count(), pruned_indexes_.gpu_data(), blobs_[0]->gpu_data(), blobs_[0]->mutable_gpu_data());
	//const unsigned int* pruned_indexes = pruned_indexes_.cpu_data();
	//Dtype* weight_data = blobs_[0]->mutable_gpu_data();
	//for (int i = 0; i < blobs_[0]->count(); i++)
	//{
	//	weight_data[i] = pruned_indexes[i] ? Dtype(0) : weight_data[i];
	//}
	ConvolutionLayer::Forward_gpu(bottom, top);
}

template <typename Dtype>
void PrunedConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!pruned_setup_)
	{
		PrunedSetUp();
	}

	ConvolutionLayer::Backward_gpu(top, propagate_down, bottom);
	//const unsigned int* pruned_indexes = pruned_indexes_.cpu_data();
	//Dtype* weight_diff = blobs_[0]->mutable_gpu_diff();
	//for (int i = 0; i < blobs_[0]->count(); i++)
	//{
	//	weight_diff[i] = pruned_indexes[i] ? 0 : weight_diff[i];
	//}
	caffe_gpu_mul(blobs_[0]->count(), pruned_indexes_.gpu_data(), blobs_[0]->gpu_diff(), blobs_[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(PrunedConvolutionLayer);

}  // namespace caffe
