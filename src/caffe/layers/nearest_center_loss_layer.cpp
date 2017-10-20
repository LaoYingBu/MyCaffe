#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/nearest_center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NearestCenterLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.center_loss_param().num_output();  
  N_ = num_output;
  //const int axis = bottom[0]->CanonicalAxisIndex(
  //    this->layer_param_.center_loss_param().axis());
	// axis is fixed to 1

  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
	M_ = bottom[0]->num();
  K_ = bottom[0]->count(1);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
	}
	else {
		this->blobs_.resize(1);
		// Intialize the weight
		vector<int> center_shape(2);
		center_shape[0] = N_;
		center_shape[1] = K_;
		this->blobs_[0].reset(new Blob<Dtype>(center_shape));
		// fill the weights
		shared_ptr<Filler<Dtype> > center_filler(GetFiller<Dtype>(
			this->layer_param_.center_loss_param().center_filler()));
		center_filler->Fill(this->blobs_[0].get());
	}  // parameter initialization
	LayerParameter softmax_loss_param(this->layer_param_);
	softmax_loss_param.set_type("SoftmaxWithLoss");
	softmax_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_loss_param);

	vector<int> softmax_shape(2);
	softmax_shape[0] = M_;
	softmax_shape[1] = N_;
	softmax_input_.Reshape(softmax_shape);
	softmax_bottom_vec_.clear();
	softmax_bottom_vec_.push_back(&softmax_input_);
	softmax_bottom_vec_.push_back(bottom[1]);
	softmax_loss_layer_->SetUp(softmax_bottom_vec_, top);
  
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void NearestCenterLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);  
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
	vector<int> dist_shape(1, K_);
  distance_.Reshape(dist_shape);
}

template <typename Dtype>
void NearestCenterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* center = this->blobs_[0]->cpu_data();
	const Dtype lamda = this->layer_param_.center_loss_param().lamda();
  Dtype* distance_data = distance_.mutable_cpu_data();
	Dtype* softmax_data = softmax_input_.mutable_cpu_data();
  
  // the i-th distance_data
  for (int m = 0; m < M_; m++) {		
		for (int n = 0; n < N_; n++) {			
			// D2(m,n) = sumsq((X(m,:) - C(n,:))			
			caffe_sub(K_, bottom_data + m * K_, center + n * K_, distance_data);
			softmax_data[m * N_ + n] = caffe_cpu_dot(K_, distance_.cpu_data(), distance_.cpu_data()) / Dtype(-2 * K_);
		}
  }
	for (int m = 0; m < M_; m++)
	{
		const int label_value = static_cast<int>(label[m]);
		softmax_data[m * N_ + label_value] *= (lamda * lamda + 1);
	}
	softmax_loss_layer_->Forward(softmax_bottom_vec_, top);
}

template <typename Dtype>
void NearestCenterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	const Dtype* label = bottom[1]->cpu_data();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* center_data = this->blobs_[0]->cpu_data();
  // Gradient with respect to centers
  if (this->param_propagate_down_[0]) {
    Dtype* center_diff = this->blobs_[0]->mutable_cpu_diff();		
		
    // \sum_{y_i==j}
		for (int n = 0; n < N_; n++) {
      int count = 0;
      for (int m = 0; m < M_; m++) {
        const int label_value = static_cast<int>(label[m]);
        if (label_value == n) {
          count++;
					caffe_sub(K_, center_diff + n * K_, bottom_data + m * K_, center_diff + n * K_);
        }
      }
			caffe_axpy(K_, (Dtype)count, center_data + n * K_, center_diff + n * K_);
      caffe_scal(K_, (Dtype)1./(count + (Dtype)1.), center_diff + n * K_);
    }
  }
  // Gradient with respect to bottom data 
  if (propagate_down[0]) {
		const Dtype lamda = this->layer_param_.center_loss_param().lamda();		
		const Dtype* softmax_diff = softmax_input_.cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();		
		Dtype* distance_data = distance_.mutable_cpu_data();

		softmax_loss_layer_->Backward(top, propagate_down, softmax_bottom_vec_);
		for (int m = 0; m < M_; m++) {
			const int label_value = static_cast<int>(label[m]);
			for (int n = 0; n < N_; n++) {
				caffe_sub(K_, bottom_data + m * K_, center_data + n * K_, distance_data);
				Dtype alpha = softmax_diff[m * N_ + n] / Dtype(K_) * ((label_value == n) ? -(lamda * lamda + 1) : -1);
				caffe_axpy(K_, alpha, distance_.cpu_data(), bottom_diff + m * K_);
			}
		}
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(NearestCenterLossLayer);
#endif

INSTANTIATE_CLASS(NearestCenterLossLayer);
REGISTER_LAYER_CLASS(NearestCenterLoss);

}  // namespace caffe
