#include <algorithm>
#include <vector>

#include "caffe/layers/soft_triplet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftTripletLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);  
	CHECK_EQ(bottom[0]->num() % 3, 0) << this->type() << " must input features from TripletDataLayer.";
	int batch_size = bottom[0]->num() / 3;
	int dim = bottom[0]->count(1);

	vector<int> dist_shape(2);
	dist_shape[0] = batch_size;
	dist_shape[1] = dim;
	pos_dist_.Reshape(dist_shape);
	neg1_dist_.Reshape(dist_shape);
	neg2_dist_.Reshape(dist_shape);
	vector<int> lost_shape(1, batch_size * 2);
	vec_lost_.Reshape(lost_shape);
}

template <typename Dtype>
void SoftTripletLossLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {	
	Dtype margin = this->layer_param_.contrastive_loss_param().margin();
	int batch_size = bottom[0]->num() / 3;
	int dim = bottom[0]->count(1);

	const Dtype* pos1_data = bottom[0]->cpu_data();
	const Dtype* pos2_data = bottom[0]->cpu_data() + batch_size * dim;
	const Dtype* neg_data = bottom[0]->cpu_data() + batch_size * 2 * dim;
	Dtype* pos_dist = pos_dist_.mutable_cpu_data();
	Dtype* neg1_dist = neg1_dist_.mutable_cpu_data();
	Dtype* neg2_dist = neg2_dist_.mutable_cpu_data();
	Dtype* vec_lost = vec_lost_.mutable_cpu_data();

	caffe_sub(batch_size * dim, pos2_data, pos1_data, pos_dist);
	caffe_sub(batch_size * dim, neg_data, pos1_data, neg1_dist);
	caffe_sub(batch_size * dim, neg_data, pos2_data, neg2_dist);
	for (int i = 0; i < batch_size; i++)
	{
		Dtype dp = caffe_cpu_dot(dim, pos_dist + i * dim, pos_dist + i * dim);
		Dtype dn1 = caffe_cpu_dot(dim, neg1_dist + i * dim, neg1_dist + i * dim);
		Dtype dn2 = caffe_cpu_dot(dim, neg2_dist + i * dim, neg2_dist + i * dim);
		vec_lost[i] = log(1 + exp((dp - dn1) / margin));
		vec_lost[batch_size + i] = log(1 + exp((dp - dn2) / margin));
	}

	Dtype loss = caffe_cpu_asum(vec_lost_.count(), vec_lost_.cpu_data());

	top[0]->mutable_cpu_data()[0] = loss / 2 / batch_size;
}

template <typename Dtype>
void SoftTripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	int batch_size = bottom[0]->num() / 3;
	int dim = bottom[0]->count(1);
	Dtype margin = this->layer_param_.contrastive_loss_param().margin();
	const Dtype* pos_dist = pos_dist_.cpu_data();
	const Dtype* neg1_dist = neg1_dist_.cpu_data();
	const Dtype* neg2_dist = neg2_dist_.cpu_data();
	const Dtype* vec_lost = vec_lost_.cpu_data();

	if (propagate_down[0]) {
		Dtype* pos1_diff = bottom[0]->mutable_cpu_diff();
		Dtype* pos2_diff = bottom[0]->mutable_cpu_diff() + batch_size * dim;
		Dtype* neg_diff = bottom[0]->mutable_cpu_diff() + batch_size * 2 * dim;

		for (int i = 0; i < batch_size; i++)
		{
			if (vec_lost[i] > 0)
			{
				Dtype c = (1 - 1 / exp(vec_lost[i])) / margin;
				caffe_axpy(dim, Dtype(c), neg2_dist + i * dim, pos1_diff + i * dim);
				caffe_axpy(dim, Dtype(c), pos_dist + i * dim, pos2_diff + i * dim);
				caffe_axpy(dim, Dtype(-c), neg1_dist + i * dim, neg_diff + i * dim);
			}

			if (vec_lost[batch_size + i] > 0)
			{
				Dtype c = (1 - 1 / exp(vec_lost[batch_size + i])) / margin;
				caffe_axpy(dim, Dtype(-c), pos_dist + i * dim, pos1_diff + i * dim);
				caffe_axpy(dim, Dtype(c), neg1_dist + i * dim, pos2_diff + i * dim);
				caffe_axpy(dim, Dtype(-c), neg2_dist + i * dim, neg_diff + i * dim);
			}
		}

		const Dtype alpha = top[0]->cpu_diff()[0] / batch_size;
		caffe_scal(bottom[0]->count(), alpha, bottom[0]->mutable_cpu_diff());
	}
}

#ifdef CPU_ONLY
STUB_GPU(SoftTripletLossLayer);
#endif

INSTANTIATE_CLASS(SoftTripletLossLayer);
REGISTER_LAYER_CLASS(SoftTripletLoss);

}  // namespace caffe
