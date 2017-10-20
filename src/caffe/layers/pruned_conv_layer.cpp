#include <vector>

#include "caffe/layers/pruned_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PrunedConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	ConvolutionLayer::LayerSetUp(bottom, top);
	pruned_setup_ = false;
}

template <typename Dtype>
void PrunedConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	ConvolutionLayer::Reshape(bottom, top);
	vector<int> shape(blobs_[0]->shape());
	pruned_indexes_.Reshape(shape);
	pruned_setup_ = false;
}

template <typename Dtype>
inline void PrunedConvolutionLayer<Dtype>::PrunedSetUp() {
	const Dtype* weight_data = blobs_[0]->cpu_data();
	Dtype* pruned_indexes = pruned_indexes_.mutable_cpu_data();	
	for (int i = 0; i < blobs_[0]->count(); i++)
	{
		pruned_indexes[i] = weight_data[i] == 0 ? Dtype(0) : Dtype(1);
	}
	pruned_setup_ = true;
}

template <typename Dtype>
void PrunedConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	if (!pruned_setup_)
	{
		PrunedSetUp();
	}
	caffe_mul(blobs_[0]->count(), pruned_indexes_.cpu_data(), blobs_[0]->cpu_data(), blobs_[0]->mutable_cpu_data());
	//const Dtype* pruned_indexes = pruned_indexes_.cpu_data();
	//Dtype* weight_data = blobs_[0]->mutable_cpu_data();
	//for (int i = 0; i < blobs_[0]->count(); i++)
	//{
	//	weight_data[i] = pruned_indexes[i] ? 0 : weight_data[i];
	//}
  ConvolutionLayer::Forward_cpu(bottom, top);
}

template <typename Dtype>
void PrunedConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!pruned_setup_)
	{
		PrunedSetUp();
	}

	ConvolutionLayer::Backward_cpu(top, propagate_down, bottom);
	//const Dtype* pruned_indexes = pruned_indexes_.cpu_data();
	//Dtype* weight_diff = blobs_[0]->mutable_cpu_diff();
	//for (int i = 0; i < blobs_[0]->count(); i++)
	//{
	//	weight_diff[i] = pruned_indexes[i] ? 0 : weight_diff[i];
	//}	
	caffe_mul(blobs_[0]->count(), pruned_indexes_.cpu_data(), blobs_[0]->cpu_diff(), blobs_[0]->mutable_cpu_diff());
}

#ifdef CPU_ONLY
STUB_GPU(PrunedConvolutionLayer);
#endif

INSTANTIATE_CLASS(PrunedConvolutionLayer);
REGISTER_LAYER_CLASS(PrunedConvolution);

}  // namespace caffe
