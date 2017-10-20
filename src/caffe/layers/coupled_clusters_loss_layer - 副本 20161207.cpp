#include <algorithm>
#include <vector>

#include "caffe/layers/coupled_clusters_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CoupledClustersLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->num() % 2, 0);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
	int batch_size = bottom[0]->num() / 2;
	int dim = bottom[0]->channels();
	vector<int> shape(1, dim);
	center_pos_.Reshape(shape);	
	vec_dist_.Reshape(bottom[0]->shape());
	vector<int> mul_shape(1, batch_size);
	mean_multiplier_.Reshape(mul_shape);
	caffe_set(batch_size, Dtype(1), mean_multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void CoupledClustersLossLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	Dtype margin = this->layer_param_.contrastive_loss_param().margin();
	int batch_size = bottom[0]->num() / 2;
	int dim = bottom[0]->channels();
	const Dtype* pos_data = bottom[0]->cpu_data();
	const Dtype* neg_data = bottom[0]->cpu_data() + batch_size * dim;
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, dim, batch_size, Dtype(1.0) / batch_size, mean_multiplier_.cpu_data(), pos_data, 0, center_pos_.mutable_cpu_data());
	//for (int i = 0; i < batch_size * dim * 2; i++)
	//	LOG(INFO) << "bottom[" << i << "]: " << bottom[0]->cpu_data()[i];
	//for (int i = 0; i < dim; i++)
	//	LOG(INFO) << "center[" << i << "]: " << center_pos_.cpu_data()[i];
	for (int item = 0; item < batch_size * 2; item++)
	{
		caffe_sub(dim, pos_data + dim * item, center_pos_.cpu_data(), vec_dist_.mutable_cpu_data() + dim * item);
	}
	//for (int i = 0; i < batch_size * dim * 2; i++)
	//	LOG(INFO) << "sub[" << i << "]: " << vec_dist_.cpu_data()[i];

	Dtype nearest_neg_dist = FLT_MAX;
	nearest_neg_index_ = 0;
	for (int i = 0; i < batch_size; i++)
	{
		const Dtype* neg_dist = vec_dist_.cpu_data() + batch_size * dim;
		Dtype dist = caffe_cpu_dot(dim, neg_dist + dim * i, neg_dist + dim * i);
		if (dist < nearest_neg_dist)
		{
			nearest_neg_index_ = i;
			nearest_neg_dist = dist;
		}		
	}

	Dtype loss(0.0);
	caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
	Dtype* pos_diff = bottom[0]->mutable_cpu_diff();
	Dtype* neg_diff = bottom[0]->mutable_cpu_diff() + batch_size * dim;
	for (int i = 0; i < batch_size; i++)
	{
		const Dtype* pos_dist = vec_dist_.cpu_data();
		Dtype dist = caffe_cpu_dot(dim, pos_dist + dim * i, pos_dist + dim * i);
		Dtype c = dist + margin - nearest_neg_dist;
		if (c > 0)
		{
			caffe_axpy(dim, Dtype(-1), vec_dist_.cpu_data() + (batch_size + nearest_neg_index_) * dim, neg_diff + nearest_neg_index_ * dim);
			caffe_axpy(dim, Dtype(1), vec_dist_.cpu_data() + i * dim, pos_diff + i * dim);
			loss += c;
		}

		//Dtype c = sqrt(1 + dist) - log(exp(sqrt(1 + dist)) + exp(sqrt(1 + nearest_neg_dist)));
		//caffe_axpy(dim, (c - 1) / sqrt(1 + nearest_neg_dist), vec_dist_.cpu_data() + (batch_size + nearest_neg_index_) * dim, neg_diff + nearest_neg_index_ * dim);
		//caffe_axpy(dim, (1 - c) / sqrt(1 + dist), vec_dist_.cpu_data() + i * dim, pos_diff + i * dim);
		//loss += log(c);
		//LOG(INFO) << "dist: " << dist << " nearest_neg_dist: " << nearest_neg_dist;
		//LOG(INFO) << "c: " << c << " log(c): " << log(c);
	}

  loss = loss / batch_size;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CoupledClustersLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	int batch_size = bottom[0]->num() / 2;
	int dim = bottom[0]->channels();
	if (propagate_down[0]) {
		const Dtype alpha = top[0]->cpu_diff()[0] / batch_size;
		caffe_scal(bottom[0]->count(), alpha, bottom[0]->mutable_cpu_diff());
	}
}

#ifdef CPU_ONLY
STUB_GPU(CoupledClustersLossLayer);
#endif

INSTANTIATE_CLASS(CoupledClustersLossLayer);
REGISTER_LAYER_CLASS(CoupledClustersLoss);

}  // namespace caffe
