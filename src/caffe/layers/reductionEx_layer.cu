#include <vector>

#include "caffe/layers/reductionEx_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ReductionExLayer<Dtype>::Forward_gpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* mult_data = NULL;
	if (sum_multiplier_.count() > 0) {
		mult_data = sum_multiplier_.gpu_data();
	}
	Dtype* top_data = top[0]->mutable_gpu_data();
	for (int i = 0; i < outer_dim_; ++i) {
		switch (op_) {
		case ReductionParameter_ReductionOp_SUM:
		case ReductionParameter_ReductionOp_MEAN:
			caffe_gpu_gemv<Dtype>(CblasTrans, reduct_dim_, inner_dim_, 1.,
				bottom_data, mult_data, 0., top_data);
			break;
		default:
			LOG(FATAL) << "Unknown reduction op: "
				<< ReductionParameter_ReductionOp_Name(op_);
		}
		bottom_data += inner_dim_ * reduct_dim_;
		top_data += inner_dim_;
	}
	if (coeff_ != Dtype(1)) {
		// Reset the top_data pointer.
		top_data = top[0]->mutable_gpu_data();
		caffe_gpu_scal(outer_dim_ * inner_dim_, coeff_, top_data);
	}
}

template <typename Dtype>
void ReductionExLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) { return; }
	// Get bottom_data, if needed.
	switch (op_) {
		// Operations that don't need bottom_data
		case ReductionParameter_ReductionOp_SUM:
		case ReductionParameter_ReductionOp_MEAN:
			break;
		default:
			LOG(FATAL) << "Unknown reduction op: "
				<< ReductionParameter_ReductionOp_Name(op_);
	}
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const Dtype* mult_data = NULL;
	if (sum_multiplier_.count() > 0) {
		mult_data = sum_multiplier_.gpu_data();
	}
	for (int i = 0; i < outer_dim_; ++i) {
		//const Dtype bottom_coeff = (*top_diff) * coeff_;
		switch (op_) {
		case ReductionParameter_ReductionOp_SUM:
		case ReductionParameter_ReductionOp_MEAN:
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, reduct_dim_, inner_dim_, 1.,
				coeff_, mult_data, top_diff, 0., bottom_diff);
			break;
		default:
			LOG(FATAL) << "Unknown reduction op: "
				<< ReductionParameter_ReductionOp_Name(op_);
		}
		bottom_diff += inner_dim_ * reduct_dim_;
		top_diff += inner_dim_;
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(ReductionExLayer);

}  // namespace caffe
