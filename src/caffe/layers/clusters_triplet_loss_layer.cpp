#include <algorithm>
#include <vector>

#include "caffe/layers/clusters_triplet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ClustersTripletLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
	const int batch_size = bottom[0]->num();
	const int count_in_cluster = this->layer_param_.clusters_data_param().count_in_cluster();
	CHECK_GT(count_in_cluster, 0) << "Clusters data count_in_cluster must be > 0";
	CHECK_EQ(batch_size % count_in_cluster, 0) << "Input of clusters triplet loss layer must be features from ClustersDataLayer.";
	const int clusters_in_batch = batch_size / count_in_cluster;
	//const Dtype* label = bottom[1]->cpu_data();
	//for (int i = 0;i < clusters_in_batch;i++)
	//{
	//	for (int j = 0; j < i; j++)
	//	{
	//		CHECK(((i / count_in_cluster == j / count_in_cluster) && (label[i] == label[j])) ||
	//			((i / count_in_cluster != j / count_in_cluster) && (label[i] != label[j]))) <<
	//			"Input of clusters triplet loss layer must be features from ClustersDataLayer.";
	//	}
	//}
	vector<int> dots_shape(1, batch_size);
	dots_.Reshape(dots_shape);
	vector<int> dists_shape(2, batch_size);
	dists_.Reshape(dists_shape);
	vector<int> coeff_shape(2, batch_size);
	vec_diff_coeff_.Reshape(coeff_shape);
}

template <typename Dtype>
void ClustersTripletLossLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {	
	Dtype margin = this->layer_param_.contrastive_loss_param().margin();
	const int batch_size = bottom[0]->num();
	const int dim = bottom[0]->count(1);
	const int count_in_cluster = this->layer_param_.clusters_data_param().count_in_cluster();
	const int clusters_in_batch = batch_size / count_in_cluster;
	
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* dot_data = dots_.mutable_cpu_data();
	Dtype* dist_data = dists_.mutable_cpu_data();
	Dtype* vec_diff_coeff = vec_diff_coeff_.mutable_cpu_data();
	for (int i = 0; i < batch_size; i++)
	{
		dot_data[i] = caffe_cpu_dot(dim, bottom_data + i * dim, bottom_data + i * dim);
	}
	for (int i = 0; i < batch_size; i++)
	{
		for (int j = 0; j < batch_size; j++)
		{
			if (i == j)
			{
				dist_data[i * batch_size + j] = 0;
			}
			else
			{
				dist_data[i * batch_size + j] = dots_.cpu_data()[i] + dots_.cpu_data()[j] - 2 * caffe_cpu_dot(dim, bottom_data + dim * i, bottom_data + dim * j);
			}
		}
	}

	Dtype loss(0);
	int valid_triplet = 0;
	caffe_set(vec_diff_coeff_.count(), Dtype(0), vec_diff_coeff_.mutable_cpu_data());
	for (int i = 0; i < batch_size; i++)
	{
		int aid = i;
		for (int j = 0; j < count_in_cluster; j++)
		{
			int pid = i - i % count_in_cluster + j;
			//if (aid == pid) continue;
			for (int k = 0; k < batch_size - count_in_cluster; k++)
			{
				int nid = k + ((k >= i - i % count_in_cluster) ? count_in_cluster : 0);

				//int ind = (i * ncount + j) * ncount + k;
				Dtype dloss = dists_.cpu_data()[aid * batch_size + pid] + margin - dists_.cpu_data()[aid * batch_size + nid];
				if (dloss > 0)
				{
					vec_diff_coeff[aid * batch_size + pid] -= 1;
					vec_diff_coeff[aid * batch_size + nid] += 1;
					vec_diff_coeff[pid * batch_size + pid] += 1;
					vec_diff_coeff[pid * batch_size + aid] -= 1;
					vec_diff_coeff[nid * batch_size + aid] += 1;
					vec_diff_coeff[nid * batch_size + nid] -= 1;

					loss += dloss;
				}
				valid_triplet++;
			}
		}
	}

	top[0]->mutable_cpu_data()[0] = loss / 2 / valid_triplet;
	caffe_scal(vec_diff_coeff_.count(), Dtype(1) / valid_triplet, vec_diff_coeff_.mutable_cpu_data());
}

template <typename Dtype>
void ClustersTripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
STUB_GPU(ClustersTripletLossLayer);
#endif

INSTANTIATE_CLASS(ClustersTripletLossLayer);
REGISTER_LAYER_CLASS(ClustersTripletLoss);

}  // namespace caffe
