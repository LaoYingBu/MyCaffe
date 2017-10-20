#ifndef CAFFE_GRID_TRIPLET_LOSS_LAYER_HPP_
#define CAFFE_GRID_TRIPLET_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the triplet loss across channel @f$
 *          d_pos = L2{pos_1 - pos_2}
 *          d_neg1 = L2{neg - pos_1}
 *          d_neg2 = L2{neg - pos_2}
 *          loss = max(d_pos + margin - d_neg1, 0) + max(d_pos + margin - d_neg1, 0)
 *
 */
template <typename Dtype>
class GridTripletLossLayer : public LossLayer<Dtype> {
public:
	explicit GridTripletLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline const char* type() const { return "GridTripletLoss"; }

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 2;
  }

 protected:
  /// @copydoc TripletLossLayer
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
	Blob<Dtype> dot_vec_;
	Blob<Dtype> pos_sum_dist_;
	Blob<Dtype> neg1_sum_dist_;
	Blob<Dtype> neg2_sum_dist_;
	Blob<Dtype> sum_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_CLUSTERS_TRIPLET_LOSS_LAYER_HPP_
