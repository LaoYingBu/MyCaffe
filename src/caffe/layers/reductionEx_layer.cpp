#include <vector>

#include "caffe/layers/reductionEx_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ReductionExLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  op_ = this->layer_param_.reduction_param().operation();
}

template <typename Dtype>
void ReductionExLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  start_axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.reduction_param().axis());
	end_axis_ = bottom[0]->CanonicalAxisIndex(
		this->layer_param_.reduction_param().end_axis());
  // In the output, we'll keep all axes up to the reduction axis
  // we'd need to also copy any axes following an "end_axis".
	vector<int> top_shape;
	for (int i = 0; i < start_axis_; ++i) {
		top_shape.push_back(bottom[0]->shape(i));
	}	
	top_shape.push_back(1);
	for (int i = end_axis_ + 1; i < bottom[0]->num_axes(); ++i) {
		top_shape.push_back(bottom[0]->shape(i));
	}

  top[0]->Reshape(top_shape);
	outer_dim_ = bottom[0]->count(0, start_axis_);
	reduct_dim_ = bottom[0]->count(start_axis_, end_axis_ + 1);
	inner_dim_ = bottom[0]->count(end_axis_ + 1, bottom[0]->num_axes());

  CHECK_EQ(outer_dim_ * inner_dim_, top[0]->count());
	CHECK(op_ == ReductionParameter_ReductionOp_SUM ||
		op_ == ReductionParameter_ReductionOp_MEAN) << "Only Sum or Mean operation of reductionEx now.";
  if (op_ == ReductionParameter_ReductionOp_SUM ||
      op_ == ReductionParameter_ReductionOp_MEAN) {
    vector<int> sum_mult_shape(1, reduct_dim_);
    sum_multiplier_.Reshape(sum_mult_shape);
		caffe_set(reduct_dim_, Dtype(1), sum_multiplier_.mutable_cpu_data());
  }

	coeff_ = this->layer_param().reduction_param().coeff();
  if (op_ == ReductionParameter_ReductionOp_MEAN) {
		coeff_ /= reduct_dim_;
  }
}

template <typename Dtype>
void ReductionExLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* mult_data = NULL;
  if (sum_multiplier_.count() > 0) {
    mult_data = sum_multiplier_.cpu_data();
  }
  Dtype* top_data = top[0]->mutable_cpu_data();
	for (int i = 0; i < outer_dim_; ++i) {
		switch (op_) {
		case ReductionParameter_ReductionOp_SUM:
		case ReductionParameter_ReductionOp_MEAN:
			caffe_cpu_gemv<Dtype>(CblasTrans, reduct_dim_, inner_dim_, 1.,
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
    top_data = top[0]->mutable_cpu_data();
    caffe_scal(outer_dim_ * inner_dim_, coeff_, top_data);
  }
}

template <typename Dtype>
void ReductionExLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* mult_data = NULL;
	if (sum_multiplier_.count() > 0) {
		mult_data = sum_multiplier_.cpu_data();
	}
  for (int i = 0; i < outer_dim_; ++i) {
    //const Dtype bottom_coeff = (*top_diff) * coeff_;
    switch (op_) {
    case ReductionParameter_ReductionOp_SUM:
    case ReductionParameter_ReductionOp_MEAN:
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, reduct_dim_, inner_dim_, 1.,
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

#ifdef CPU_ONLY
STUB_GPU(ReductionExLayer);
#endif

INSTANTIATE_CLASS(ReductionExLayer);
REGISTER_LAYER_CLASS(ReductionEx);

}  // namespace caffe
