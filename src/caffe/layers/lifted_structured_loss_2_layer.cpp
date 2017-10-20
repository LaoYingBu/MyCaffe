#include <algorithm>
#include <vector>

#include "caffe/layers/lifted_structured_loss_2_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LiftedStructuredLoss2Layer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
	const int batch_size = bottom[0]->num();
	const int dim = bottom[0]->count(1);
	const int cluster_size = this->layer_param_.clusters_data_param().cluster_size();
	CHECK_GT(cluster_size, 0) << "Clusters data cluster_size must be > 0";
	CHECK_EQ(batch_size % cluster_size, 0) << "Input of clusters triplet loss layer must be features from ClustersDataLayer.";
	const int num_clusters = batch_size / cluster_size;
	const Dtype* label = bottom[1]->cpu_data();
	//for (int i = 0;i < num_clusters;i++)
	//{
	//	for (int j = 0; j < i; j++)
	//	{
	//		CHECK(((i / cluster_size == j / cluster_size) && (label[i] == label[j])) ||
	//			((i / cluster_size != j / cluster_size) && (label[i] != label[j]))) <<
	//			"Input of clusters triplet loss layer must be features from ClustersDataLayer.";
	//	}
	//}
	sqrs_.ReshapeLike(*(bottom[0]));
	vector<int> dots_shape(1, batch_size);
	dots_.Reshape(dots_shape);
	//exps_.Reshape(dots_shape);
	vector<int> dists_shape(2, batch_size);
	dists_.Reshape(dists_shape);
	temp_.Reshape(dists_shape);	
	vec_diff_coeff_.Reshape(dists_shape);
	vector<int> mult_shape(1, std::max(dim, batch_size));
	sum_multiplier_.Reshape(mult_shape);	
	caffe_set(sum_multiplier_.count(), Dtype(1), sum_multiplier_.mutable_cpu_data());	
	vector<int> weight_shape(1, 1);
	this->blobs_.resize(1);
	this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
	this->blobs_[0]->mutable_cpu_data()[0] = Dtype(1.07);
}

template <typename Dtype>
void LiftedStructuredLoss2Layer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {	
	Dtype margin = this->layer_param_.contrastive_loss_param().margin();
	const int batch_size = bottom[0]->num();
	const int dim = bottom[0]->count(1);
	const int cluster_size = this->layer_param_.clusters_data_param().cluster_size();
	const int num_clusters = batch_size / cluster_size;
	
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* dot_data = dots_.mutable_cpu_data();
	Dtype* dist_data = dists_.mutable_cpu_data();
	Dtype* vec_diff_coeff = vec_diff_coeff_.mutable_cpu_data();
	Dtype* sqr_data = sqrs_.mutable_cpu_data();
	Dtype* temp_data = temp_.mutable_cpu_data();
	Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();	
	//Dtype* exp_data = exps_.mutable_cpu_data();

	caffe_powx(bottom[0]->count(), bottom_data, Dtype(2), sqr_data);
	caffe_cpu_gemv(CblasNoTrans, batch_size, dim, Dtype(1), sqr_data, multiplier_data, Dtype(0), dot_data);
	caffe_cpu_gemm(CblasNoTrans, CblasTrans, batch_size, batch_size, dim, Dtype(-2), bottom_data, bottom_data, Dtype(0), dist_data);
	caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, batch_size, batch_size, 1, Dtype(1), dot_data, multiplier_data, Dtype(0), temp_data);
	caffe_axpy(dists_.count(), Dtype(1), temp_data, dist_data);
	caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, batch_size, batch_size, 1, Dtype(1), multiplier_data, dot_data, Dtype(0), temp_data);
	caffe_axpy(dists_.count(), Dtype(1), temp_data, dist_data);

	//for (int i = 0; i < dists_.count(); i++)
	//	LOG(INFO) << dist_data[i];

	Dtype loss(0);
	caffe_set(vec_diff_coeff_.count(), Dtype(0), vec_diff_coeff_.mutable_cpu_data());
	//for (int di = 0; di < batch_size; di++)
	//{
	//	int c = di / cluster_size;
	//	Dtype sum_exp = 0;
	//	for (int k = 0; k < batch_size; k++)
	//	{
	//		if (k / cluster_size == c) continue;
	//		sum_exp += exp(margin - dist_data[di * batch_size + k]);
	//	}
	//	exp_data[di] = std::max(sum_exp, Dtype(FLT_MIN));
	//}
	
	Dtype theta = std::max(this->blobs_[0]->cpu_data()[0], Dtype(FLT_MIN));
	Dtype dtheta = 0;
	for (int di = 0; di < batch_size; di++)
	{
		int c = di / cluster_size;		
		
		Dtype min_dist_n = Dtype(FLT_MAX);
		Dtype max_dist_p = Dtype(-FLT_MAX);
		for (int k = 0; k < batch_size; k++)
		{
			if (k / cluster_size == c)
			{
				max_dist_p = std::max(max_dist_p, dist_data[di * batch_size + k]);
			}
			else
			{
				min_dist_n = std::min(min_dist_n, dist_data[di * batch_size + k]);
			}
		}

		Dtype sum_exp_n = 0;
		Dtype sum_exp_p = 0;
		for (int k = 0; k < batch_size; k++)
		{
			if (k / cluster_size == c) 
			{
				sum_exp_p += exp(theta * (dist_data[di * batch_size + k] - max_dist_p));
			}
			else
			{
				sum_exp_n += exp(theta * (margin + min_dist_n - dist_data[di * batch_size + k]));
			}
		}

		Dtype lossk = log(sum_exp_n) / theta - min_dist_n + log(sum_exp_p) / theta + max_dist_p;		
		if (lossk > 0)
		{
			loss += lossk;
			dtheta -= lossk / theta;
			for (int k = 0; k < batch_size; k++)
			{
				if (k / cluster_size == c) 
				{
					Dtype d = exp(theta * (dist_data[di * batch_size + k] - max_dist_p)) / sum_exp_p;
					vec_diff_coeff[di * batch_size + di] += d;
					vec_diff_coeff[di * batch_size + k] -= d;
					vec_diff_coeff[k * batch_size + k] += d;
					vec_diff_coeff[k * batch_size + di] -= d;
					dtheta += (dist_data[di * batch_size + k]) * d / theta;
				}
				else
				{
					Dtype d = exp(theta * (margin + min_dist_n - dist_data[di * batch_size + k])) / sum_exp_n;
					vec_diff_coeff[di * batch_size + di] -= d;
					vec_diff_coeff[di * batch_size + k] += d;
					vec_diff_coeff[k * batch_size + k] -= d;
					vec_diff_coeff[k * batch_size + di] += d;
					dtheta += (margin - dist_data[di * batch_size + k]) * d / theta;
				}
			}
		}
	}

	int valid_triplet = batch_size;
	top[0]->mutable_cpu_data()[0] = loss / 2 / valid_triplet;
	this->blobs_[0]->mutable_cpu_diff()[0] = dtheta / valid_triplet;
	caffe_scal(vec_diff_coeff_.count(), Dtype(1) / valid_triplet, vec_diff_coeff_.mutable_cpu_data());
}

template <typename Dtype>
void LiftedStructuredLoss2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const int batch_size = bottom[0]->num();
	const int dim = bottom[0]->count(1);

	if (propagate_down[0]) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* vec_diff_coeff = vec_diff_coeff_.cpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		const Dtype alpha = top[0]->cpu_diff()[0];
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, batch_size, dim, batch_size, alpha,
			vec_diff_coeff, bottom_data, Dtype(0), bottom_diff);
	}

}

#ifdef CPU_ONLY
STUB_GPU(LiftedStructuredLoss2Layer);
#endif

INSTANTIATE_CLASS(LiftedStructuredLoss2Layer);
REGISTER_LAYER_CLASS(LiftedStructuredLoss2);

}  // namespace caffe
