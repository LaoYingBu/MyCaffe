#ifndef CAFFE_SOFT_TRIPLET_LOSS_LAYER_HPP_
#define CAFFE_SOFT_TRIPLET_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the soft triplet loss @f$
 *          d_pos = L2{pos_1 - pos_2}
 *          d_neg1 = L2{neg - pos_1}
 *          d_neg2 = L2{neg - pos_2}
 *          loss_pn = - log(exp(d_neg / margin) / (exp(d_pos / margin) + exp(d_neg / margin))
                    = log(1 + exp((d_pos - d_neg) / margin))
 *          loss = loss_pn1 + loss_pn2
 *
 */
template <typename Dtype>
class SoftTripletLossLayer : public LossLayer<Dtype> {
 public:
  explicit SoftTripletLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline const char* type() const { return "SoftTripletLoss"; }

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 2;
  }

 protected:
  /// @copydoc SoftTripletLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the Triplet error gradient w.r.t. the inputs.
   *
   */
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> pos_dist_;
	Blob<Dtype> neg1_dist_;
	Blob<Dtype> neg2_dist_;
	Blob<Dtype> vec_lost_;
};

}  // namespace caffe

#endif  // CAFFE_CLUSTERS_TRIPLET_LOSS_LAYER_HPP_
