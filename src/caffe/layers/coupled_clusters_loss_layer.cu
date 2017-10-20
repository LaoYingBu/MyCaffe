#include <algorithm>
#include <vector>

#include "caffe/layers/coupled_clusters_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void CoupledClustersLossLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		CoupledClustersLossLayer<Dtype>::Forward_cpu(bottom, top);

		//int batch_size = bottom[0]->num() / 2;
		//int dim = bottom[0]->channels();
		//const Dtype* pos_data = bottom[0]->gpu_data();
		//const Dtype* neg_data = bottom[0]->gpu_data() + batch_size * dim;
		//caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, dim, batch_size, 1, mean_multiplier_.gpu_data(), pos_data, 0, center_pos_.mutable_gpu_data());
		//for (int item = 0; item < batch_size * 2; item++)
		//{
		//	caffe_gpu_sub(dim, pos_data + dim * item, center_pos_.gpu_data(), vec_dist_.mutable_gpu_data() + dim * item);
		//}

		//Dtype nearest_neg_dist = FLT_MAX;
		//nearest_neg_index_ = 0;
		//for (int i = 0; i < batch_size; i++)
		//{
		//	const Dtype* neg_dist = vec_dist_.gpu_data() + batch_size * dim;
		//	Dtype dist;
		//	caffe_gpu_dot(dim, neg_dist + dim * i, neg_dist + dim * i, &dist);
		//	if (dist > nearest_neg_dist)
		//	{
		//		nearest_neg_index_ = i;
		//		nearest_neg_dist = dist;
		//	}
		//}

		//Dtype loss(0.0);
		//caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
		//Dtype* pos_diff = bottom[0]->mutable_gpu_diff();
		//Dtype* neg_diff = bottom[0]->mutable_gpu_diff() + batch_size * dim;
		//for (int i = 0; i < batch_size; i++)
		//{
		//	const Dtype* pos_dist = vec_dist_.gpu_data();
		//	Dtype dist;
		//	caffe_gpu_dot(dim, pos_dist + dim * i, pos_dist + dim * i, &dist);
		//	Dtype c = exp(sqrt(dist)) / (exp(sqrt(dist)) + exp(sqrt(nearest_neg_dist)));
		//	caffe_gpu_axpy(dim, c * (c - 1) / sqrt(nearest_neg_dist), vec_dist_.gpu_data() + (batch_size + nearest_neg_index_) * dim, neg_diff + nearest_neg_index_ * dim);
		//	caffe_gpu_axpy(dim, c * (1 - c) / sqrt(dist), vec_dist_.gpu_data() + i * dim, pos_diff + i * dim);
		//	loss += c;
		//}

		//loss = loss / batch_size;
		//top[0]->mutable_cpu_data()[0] = loss;
	}

	template <typename Dtype>
	void CoupledClustersLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		CoupledClustersLossLayer<Dtype>::Backward_cpu(top, propagate_down, bottom);

		//int batch_size = bottom[0]->num() / 2;
		//int dim = bottom[0]->channels();
		//if (propagate_down[0]) {
		//	const Dtype alpha = top[0]->cpu_diff()[0] / batch_size;
		//	caffe_gpu_scal(bottom[0]->count(), alpha, bottom[0]->mutable_gpu_diff());
		//}
	}

INSTANTIATE_LAYER_GPU_FUNCS(CoupledClustersLossLayer);

}  // namespace caffe
