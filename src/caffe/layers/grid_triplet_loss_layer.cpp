#include <algorithm>
#include <vector>

#include "caffe/layers/grid_triplet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GridTripletLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);  
	CHECK_EQ(bottom[0]->num() % 3, 0) << this->type() << " must input features from TripletDataLayer.";
	int batch_size = bottom[0]->num() / 3;
	int channels = bottom[0]->channels();
	int spatial_dim = bottom[0]->height() * bottom[0]->width();

	vector<int> dist_shape(bottom[0]->shape());
	dist_shape[0] = batch_size;
	pos_dist_.Reshape(dist_shape);
	neg1_dist_.Reshape(dist_shape);
	neg2_dist_.Reshape(dist_shape);
	vector<int> dot_shape(bottom[0]->shape());
	dot_shape[0] = 1;
	dot_vec_.Reshape(dot_shape);
	vector<int> sum_dist_shape(bottom[0]->shape());
	sum_dist_shape[0] = batch_size;
	sum_dist_shape[1] = 1;
	pos_sum_dist_.Reshape(sum_dist_shape);
	neg1_sum_dist_.Reshape(sum_dist_shape);
	neg2_sum_dist_.Reshape(sum_dist_shape);
	vector<int> multiplier_shape(1, bottom[0]->channels());
	sum_multiplier_.Reshape(multiplier_shape);
	caffe_set(sum_multiplier_.count(), Dtype(1), sum_multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void GridTripletLossLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {	
	Dtype margin = this->layer_param_.contrastive_loss_param().margin();
	int batch_size = bottom[0]->num() / 3;
	int channels = bottom[0]->channels();
	int spatial_dim = bottom[0]->height() * bottom[0]->width();

	const Dtype* pos1_data = bottom[0]->cpu_data();
	const Dtype* pos2_data = bottom[0]->cpu_data() + batch_size * channels * spatial_dim;
	const Dtype* neg_data = bottom[0]->cpu_data() + batch_size * 2 * channels * spatial_dim;
	Dtype* pos_dist = pos_dist_.mutable_cpu_data();
	Dtype* neg1_dist = neg1_dist_.mutable_cpu_data();
	Dtype* neg2_dist = neg2_dist_.mutable_cpu_data();
	Dtype* dot_vec = dot_vec_.mutable_cpu_data();
	Dtype* pos_sum_dist = pos_sum_dist_.mutable_cpu_data();
	Dtype* neg1_sum_dist = neg1_sum_dist_.mutable_cpu_data();
	Dtype* neg2_sum_dist = neg2_sum_dist_.mutable_cpu_data();

	caffe_sub(batch_size * channels * spatial_dim, pos2_data, pos1_data, pos_dist);
	caffe_sub(batch_size * channels * spatial_dim, neg_data, pos1_data, neg1_dist);
	caffe_sub(batch_size * channels * spatial_dim, neg_data, pos2_data, neg2_dist);

	Dtype loss = 0;

	for (int item = 0; item < batch_size; item++)
	{
		caffe_mul(channels * spatial_dim, pos_dist + item * channels * spatial_dim,
			pos_dist + item * channels * spatial_dim, dot_vec);
		caffe_cpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, 1,
			dot_vec, sum_multiplier_.cpu_data(), 0.,
			pos_sum_dist + item * spatial_dim);  // Sum(X^2)

		caffe_mul(channels * spatial_dim, neg1_dist + item * channels * spatial_dim, 
			neg1_dist + item * channels * spatial_dim, dot_vec);
		caffe_cpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, 1,
			dot_vec, sum_multiplier_.cpu_data(), 0.,
			neg1_sum_dist + item * spatial_dim);  // Sum(X^2)

		caffe_mul(channels * spatial_dim, neg2_dist + item * channels * spatial_dim, 
			neg2_dist + item * channels * spatial_dim, dot_vec);
		caffe_cpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, 1,
			dot_vec, sum_multiplier_.cpu_data(), 0.,
			neg2_sum_dist + item * spatial_dim);  // Sum(X^2)

		for (int j = 0; j < spatial_dim; j++)
		{
			int idj = item * spatial_dim + j;
			loss += max(Dtype(0), pos_sum_dist[idj] + margin - neg1_sum_dist[idj]);
			loss += max(Dtype(0), pos_sum_dist[idj] + margin - neg2_sum_dist[idj]);
		}
	}
	top[0]->mutable_cpu_data()[0] = loss / 2 / batch_size / spatial_dim;
}

template <typename Dtype>
void GridTripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	Dtype margin = this->layer_param_.contrastive_loss_param().margin();
	int batch_size = bottom[0]->num() / 3;
	int channels = bottom[0]->channels();
	int spatial_dim = bottom[0]->height() * bottom[0]->width();

	const Dtype* pos_dist = pos_dist_.cpu_data();
	const Dtype* neg1_dist = neg1_dist_.cpu_data();
	const Dtype* neg2_dist = neg2_dist_.cpu_data();
	const Dtype* pos_sum_dist = pos_sum_dist_.cpu_data();
	const Dtype* neg1_sum_dist = neg1_sum_dist_.cpu_data();
	const Dtype* neg2_sum_dist = neg2_sum_dist_.cpu_data();

	if (propagate_down[0]) {
		Dtype* pos1_diff = bottom[0]->mutable_cpu_diff();
		Dtype* pos2_diff = bottom[0]->mutable_cpu_diff() + batch_size * channels * spatial_dim;
		Dtype* neg_diff = bottom[0]->mutable_cpu_diff() + batch_size * 2 * channels * spatial_dim;

		for (int item = 0; item < batch_size; item++)
		{
			for (int j = 0; j < spatial_dim; j++)
			{
				int idj = item * spatial_dim + j;
				if (pos_sum_dist[idj] + margin - neg1_sum_dist[idj] > 0)
				{
					for (int k = 0; k < channels; k++)
					{
						int index = (item * channels + k) * spatial_dim + j;
						pos1_diff[index] += neg2_dist[index];
						pos2_diff[index] += pos_dist[index];	
						neg_diff[index] -= neg1_dist[index];
					}
				}
				if (pos_sum_dist[idj] + margin - neg2_sum_dist[idj] > 0)
				{
					for (int k = 0; k < channels; k++)
					{
						int index = (item * channels + k) * spatial_dim + j;
						pos1_diff[index] -= pos_dist[index];
						pos2_diff[index] += neg1_dist[index];
						neg_diff[index] -= neg2_dist[index];
					}
				}
			}
		}		

		const Dtype alpha = top[0]->cpu_diff()[0] / batch_size / spatial_dim;
		caffe_scal(bottom[0]->count(), alpha, bottom[0]->mutable_cpu_diff());
	}
}

#ifdef CPU_ONLY
STUB_GPU(GridTripletLossLayer);
#endif

INSTANTIATE_CLASS(GridTripletLossLayer);
REGISTER_LAYER_CLASS(GridTripletLoss);

}  // namespace caffe
