#ifndef CAFFE_COUPLED_CLUSTERS_LOSS_LAYER_HPP_
#define CAFFE_COUPLED_CLUSTERS_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the coupled clusters loss @f$
 *          c_p = mean(pos_i) / n_pos
 *          d_pos_i = L2{pos_i - c_p}
 *          d_neg = min(L2{neg_j - c_p})
 *          loss = sigma(exp(d_pos_i) / (exp(d_pos_i) + exp(d_neg))
 *
 */
template <typename Dtype>
class CoupledClustersLossLayer : public LossLayer<Dtype> {
 public:
  explicit CoupledClustersLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline const char* type() const { return "CoupledClustersLoss"; }
  /**
   * Unlike most loss layers, in the CoupledClustersLossLayer we can backpropagate
   * to the first two inputs.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 2;
  }

 protected:
  /// @copydoc CoupledClustersLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the CoupledClusters error gradient w.r.t. the inputs.
   *
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> center_pos_;  // cached for backward pass	
	Blob<Dtype> mean_multiplier_;
	Blob<Dtype> vec_dist_;
	int nearest_neg_index_;
};

}  // namespace caffe

#endif  // CAFFE_CoupledClusters_LOSS_LAYER_HPP_
