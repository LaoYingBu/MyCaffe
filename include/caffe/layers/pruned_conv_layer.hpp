#ifndef CAFFE_PRUNED_CONV_LAYER_HPP_
#define CAFFE_PRUNED_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
class PrunedConvolutionLayer : public ConvolutionLayer<Dtype> {
 public:
  explicit PrunedConvolutionLayer(const LayerParameter& param)
		: ConvolutionLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PrunedConvolution"; }

 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	void PrunedSetUp();	
	bool pruned_setup_;
	Blob<Dtype> pruned_indexes_;
};

}  // namespace caffe

#endif  // CAFFE_PRUNED_CONV_LAYER_HPP_
