#ifndef CAFFE_ROTATE_CONV_LAYER_HPP_
#define CAFFE_ROTATE_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**

 */
template <typename Dtype>
class RotateConvLayer : public Layer<Dtype> {
 public:
  explicit RotataConvLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RotateConv"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	virtual void RotateIndex(const int height, const int width, const int orient,
		int& idx0, int& hstep, int& wstep)

	bool flip_;
	int exnum_;
	/// The internal ConvolutionLayer.
	shared_ptr<ConvolutionLayer<Dtype> > convolution_layer_;
	/// convolution_input stores the output of the ConvolutionLayer.
	vector<shared_ptr<Blob<Dtype> > > convolution_inputs_;
	/// convolution_output stores the output of the ConvolutionLayer.
	vector<shared_ptr<Blob<Dtype> > > convolution_outputs_;
	/// The internal EltwiseLayer.
	shared_ptr<EltwiseLayer<Dtype> > eltwise_layer_;
	/// eltwise_input stores the output of the EltwiseLayer.
	vector<shared_ptr<Blob<Dtype> > > eltwise_inputs_;
	/// eltwise_output stores the output of the EltwiseLayer.
	vector<shared_ptr<Blob<Dtype> > > eltwise_outputs_;
	/// bottom vector holder to call the underlying eltwiseLayer::Forward
	vector<Blob<Dtype>*> convolution_bottom_vec_;
	/// top vector holder to call the underlying eltwiseLayer::Forward
	vector<Blob<Dtype>*> convolution_top_vec_;
	/// bottom vector holder to call the underlying EltwiseLayer::Forward
	vector<Blob<Dtype>*> eltwise_bottom_vec_;
	/// top vector holder to call the underlying EltwiseLayer::Forward
	vector<Blob<Dtype>*> eltwise_top_vec_;
};

}  // namespace caffe

#endif  // CAFFE_ROTATE_CONV_LAYER_HPP_
