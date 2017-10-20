#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/nearest_center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Compute_sqrdist_data_gpu(int nthreads, const int K, const int N,
	      const Dtype lamda, const Dtype* label, const Dtype* bottom_data,
	      const Dtype* center_data, Dtype* softmax_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int m = index / N;
		int n = index % N;
		const int label_value = static_cast<int>(label[m]);
		const Dtype alpha = ((label_value == n) ? -(lamda * lamda + 1) : -1) / Dtype(K);
		// D2(i,j) = sumsq((X(i,:) - C(j,:))
		softmax_data[index] = 0;
		for (int k = 0; k < K; k++) {
			Dtype d = bottom_data[m * K + k] - center_data[n * K + k];
			softmax_data[index] += d * d;
		}
		softmax_data[index] = softmax_data[index] * alpha / 2;
	}
}

template <typename Dtype>
__global__ void Compute_ncenter_diff_gpu(int nthreads, const int M, const int K, 
        const Dtype* label, const Dtype* bottom_data, const Dtype* center_data, 
        Dtype* center_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int count = 0;
    for (int m = 0; m < M; m++) {
      const int label_value = static_cast<int>(label[m]);
      if (label_value == index) {
        count++;
        for (int k = 0; k < K; k++) {
					center_diff[index * K + k] -= bottom_data[m * K + k] - center_data[index * K + k];
        }
      }
    }
    for (int k = 0; k < K; k++) {
			center_diff[index * K + k] = center_diff[index * K + k] / (count + (Dtype)1.);
    }
  }
}

template <typename Dtype>
__global__ void Compute_sqrdist_diff_gpu(int nthreads, const int K, const int N,
	      const Dtype lamda, const Dtype* label, const Dtype* bottom_data, const Dtype* center_data, 
				const Dtype* softmax_diff, Dtype* bottom_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int m = index / K;
		int k = index % K;
		const int label_value = static_cast<int>(label[m]);		
		//bottom_diff[index] = 0;
		for (int n = 0; n < N; n++) {
			Dtype alpha = softmax_diff[m * N + n] / Dtype(K) * ((label_value == n) ? -(lamda * lamda + 1) : -1);
			bottom_diff[index] += (bottom_data[m * K + k] - center_data[n * K + k]) * alpha;
		}		
	}
}

template <typename Dtype>
void NearestCenterLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int nthreads = M_ * N_;
	const Dtype lamda = this->layer_param_.center_loss_param().lamda();
  Compute_sqrdist_data_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
		      CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, N_, lamda, bottom[1]->gpu_data(), 
		      bottom[0]->gpu_data(), this->blobs_[0]->gpu_data(), softmax_input_.mutable_gpu_data());
	softmax_loss_layer_->Forward(softmax_bottom_vec_, top);
}

template <typename Dtype>
void NearestCenterLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	// Gradient with respect to centers
	if (this->param_propagate_down_[0]) {
		int nthreads = N_;
		Compute_ncenter_diff_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			      CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, K_, bottom[1]->gpu_data(), 
			      bottom[0]->gpu_data(), this->blobs_[0]->gpu_data(), this->blobs_[0]->mutable_gpu_diff());
	}

  if (propagate_down[0]) {
		softmax_loss_layer_->Backward(top, propagate_down, softmax_bottom_vec_);
		const Dtype lamda = this->layer_param_.center_loss_param().lamda();
		int nthreads = M_ * K_;
		Compute_sqrdist_diff_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			      CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, N_, lamda, bottom[1]->gpu_data(),  
			      bottom[0]->gpu_data(), this->blobs_[0]->gpu_data(), softmax_input_.gpu_diff(), 
			      bottom[0]->mutable_gpu_diff());
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(NearestCenterLossLayer);

}  // namespace caffe
