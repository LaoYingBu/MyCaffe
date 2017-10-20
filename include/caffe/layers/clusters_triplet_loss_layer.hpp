#ifndef CAFFE_CLUSTERS_TRIPLET_LOSS_LAYER_HPP_
#define CAFFE_CLUSTERS_TRIPLET_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 *
 */
template <typename Dtype>
class ClustersTripletLossLayer : public LossLayer<Dtype> {
 public:
  explicit ClustersTripletLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline const char* type() const { return "ClustersTripletLoss"; }
  /**
   * Unlike most loss layers, in the ClustersTripletLossLayer we can backpropagate
   * to the first two inputs.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 2;
  }

 protected:
  /// @copydoc ClustersTripletLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //    const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the ClustersTriplet error gradient w.r.t. the inputs.
   *
   */
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	//virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	//	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> dots_;  // b
	Blob<Dtype> dists_;  // b * b
	Blob<Dtype> vec_diff_coeff_;  // b * b
};

}  // namespace caffe

#endif  // CAFFE_CLUSTERS_TRIPLET_LOSS_LAYER_HPP_
