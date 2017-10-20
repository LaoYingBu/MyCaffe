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
	diff_.Reshape(bottom[0]->shape());
	vector<int> mul_shape(1, batch_size);
	mean_multiplier_.Reshape(mul_shape);
	vector<int> adist_shape(1, batch_size);
	adist_.Reshape(adist_shape);;
	caffe_set(batch_size, Dtype(1), mean_multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void CoupledClustersLossLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {	
	Dtype margin = this->layer_param_.contrastive_loss_param().margin();
	Dtype w_L2 = 0.001;
	int batch_size = bottom[0]->num() / 2;
	int dim = bottom[0]->channels();
	const Dtype* pos_data = bottom[0]->cpu_data();
	const Dtype* neg_data = bottom[0]->cpu_data() + batch_size * dim;
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, dim, batch_size, Dtype(1.0) / batch_size, mean_multiplier_.cpu_data(), pos_data, 0, center_pos_.mutable_cpu_data());
	//caffe_set(dim, 1 / sqrt(Dtype(dim)), center_pos_.mutable_cpu_data());
	for (int item = 0; item < batch_size * 2; item++)
	{
		caffe_sub(dim, pos_data + dim * item, center_pos_.cpu_data(), vec_dist_.mutable_cpu_data() + dim * item);
	}

	Dtype loss(0.0);
	Dtype* diff_data = diff_.mutable_cpu_data();
	caffe_set(diff_.count(), Dtype(0), diff_data);
	for (int i = 0;i < batch_size; i++)
	{
		Dtype* adist_data = adist_.mutable_cpu_data();
		const Dtype* pos_dist = vec_dist_.cpu_data();
		adist_data[i] = caffe_cpu_dot(dim, pos_dist + dim * i, pos_dist + dim * i) / 2;
		loss += adist_data[i];
		caffe_axpy(dim, Dtype((batch_size - 1)) / batch_size, vec_dist_.cpu_data() + i * dim, diff_data + i * dim);
	}

	for (int i = 0;i < batch_size; i++)
	{
		const Dtype* neg_dist = vec_dist_.cpu_data() + batch_size * dim;
		Dtype dist = caffe_cpu_dot(dim, neg_dist + dim * i, neg_dist + dim * i) / 2;
		int c = 0;
		for (int j = 0;j < batch_size; j++)
		{
			if (adist_.cpu_data()[j] + margin - dist > 0)
			{
				c++;
				loss += adist_.cpu_data()[j] + margin - dist;
			}
		}
		caffe_axpy(dim, Dtype(-c) / batch_size, neg_dist + dim * i, diff_data + (batch_size + i) * dim);
		caffe_axpy(dim, Dtype(c) / batch_size, neg_dist + dim * i, center_pos_.mutable_cpu_diff());
	}

	for (int i = 0;i < batch_size; i++)
	{
		caffe_axpy(dim, Dtype(1) / batch_size, center_pos_.cpu_diff(), diff_data + i * dim);
	}

	//Dtype nearest_neg_dist = FLT_MAX;
	//nearest_neg_index_ = -1;
	//for (int i = 0; i < batch_size; i++)
	//{
	//	const Dtype* neg_dist = vec_dist_.cpu_data() + batch_size * dim;
	//	Dtype dist = caffe_cpu_dot(dim, neg_dist + dim * i, neg_dist + dim * i) / 2;
	//	if (dist < nearest_neg_dist)
	//	{
	//		nearest_neg_index_ = i;
	//		nearest_neg_dist = dist;
	//	}
	//}
	//CHECK_GE(nearest_neg_index_, 0);

	
	//for (int i = 0; i < batch_size * 2; i++)
	//{
	//	Dtype norm = caffe_cpu_dot(dim, pos_data + i * dim, pos_data + i * dim);
	//	loss += w_L2 * norm / 2;
	//}
	//caffe_cpu_scale(bottom[0]->count(), w_L2, bottom[0]->cpu_data(), bottom[0]->mutable_cpu_diff());
	//LOG(INFO) << "loss0 = " << loss;

	//Dtype* diff_data = diff_.mutable_cpu_data();
	//caffe_set(diff_.count(), Dtype(0), diff_data);
	//for (int i = 0; i < batch_size; i++)
	//{
	//	const Dtype* pos_dist = vec_dist_.cpu_data();
	//	Dtype dist = caffe_cpu_dot(dim, pos_dist + dim * i, pos_dist + dim * i) / 2;
	//	Dtype c = dist + margin - nearest_neg_dist;
	//	if (c > 0)
	//	{
	//		loss += c;
	//		caffe_axpy(dim, Dtype(-1), vec_dist_.cpu_data() + (batch_size + nearest_neg_index_) * dim, diff_data + (batch_size + nearest_neg_index_) * dim);
	//		caffe_axpy(dim, Dtype(1), vec_dist_.cpu_data() + i * dim, diff_data + i * dim);
	//	}
	//}

  loss = loss / batch_size;
  top[0]->mutable_cpu_data()[0] = loss;
	//LOG(INFO) << "c_loss = " << loss;
}

template <typename Dtype>
void CoupledClustersLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	int batch_size = bottom[0]->num() / 2;
	int dim = bottom[0]->channels();
	if (propagate_down[0]) {
		const Dtype alpha = top[0]->cpu_diff()[0] / batch_size;
		caffe_cpu_scale(bottom[0]->count(), alpha, diff_.cpu_data(), bottom[0]->mutable_cpu_diff());
	}
}

#ifdef CPU_ONLY
STUB_GPU(CoupledClustersLossLayer);
#endif

INSTANTIATE_CLASS(CoupledClustersLossLayer);
REGISTER_LAYER_CLASS(CoupledClustersLoss);

}  // namespace caffe
