#include <algorithm>
#include <vector>

#include "caffe/layers/clusters_triplet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ClustersTripletLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);  
	CHECK_GE(bottom[0]->num_axes(), 2) << this->type() << " must have axis of 2 or 3.";
	CHECK_LE(bottom[0]->num_axes(), 3) << this->type() << " must have axis of 2 or 3.";
	CHECK_EQ(bottom[0]->shape(-2) % 2, 0) << this->type() << " must input same positive and negative datas.";
	int dim = bottom[0]->shape(-1);
	int ncount = bottom[0]->shape(-2) / 2;
	int batch_size = (bottom[0]->num_axes() == 2) ? 1 : bottom[0]->num();

	vector<int> dot_shape(1, ncount);
	pos_dot_.Reshape(dot_shape);
	neg_dot_.Reshape(dot_shape);
	vector<int> dist_shape(2, ncount);
	pos_dist_.Reshape(dist_shape);
	neg_dist_.Reshape(dist_shape);
	//vector<int> loss_shape(3, batch_size);
	//vec_loss_.Reshape(loss_shape);
	vector<int> coeff_shape(1, batch_size);
	coeff_shape.push_back(ncount * 2);
	coeff_shape.push_back(ncount * 2);
	vec_diff_coeff_.Reshape(coeff_shape);
	//diff_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void ClustersTripletLossLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {	
	Dtype margin = this->layer_param_.contrastive_loss_param().margin();
	int dim = bottom[0]->shape(-1);
	int ncount = bottom[0]->shape(-2) / 2;
	int batch_size = (bottom[0]->num_axes() == 2) ? 1 : bottom[0]->num();

	Dtype loss(0.0);
	caffe_set(vec_diff_coeff_.count(), Dtype(0), vec_diff_coeff_.mutable_cpu_data());
	for (int item = 0; item < batch_size; item++)
	{
		const Dtype* pos_data = bottom[0]->cpu_data() + (item * 2) * ncount * dim;
		const Dtype* neg_data = bottom[0]->cpu_data() + (item * 2 + 1) * ncount * dim;
		Dtype* pos_dot = pos_dot_.mutable_cpu_data();
		Dtype* neg_dot = neg_dot_.mutable_cpu_data();
		Dtype* pos_dist = pos_dist_.mutable_cpu_data();
		Dtype* neg_dist = neg_dist_.mutable_cpu_data();
		//Dtype* vec_loss = vec_loss_.mutable_cpu_data();
		Dtype* vec_diff_coeff = vec_diff_coeff_.mutable_cpu_data() + item * ncount * ncount * 4;
		for (int i = 0; i < ncount; i++)
		{
			pos_dot[i] = caffe_cpu_dot(dim, pos_data + i * dim, pos_data + i * dim);
			neg_dot[i] = caffe_cpu_dot(dim, neg_data + i * dim, neg_data + i * dim);
		}
		for (int i = 0; i < ncount; i++)
		{
			for (int j = 0; j < ncount; j++)
			{
				int ind = i * ncount + j;

				if (i == j)
				{
					pos_dist[ind] = 0;
				}
				else
				{
					pos_dist[ind] = pos_dot_.cpu_data()[i] + pos_dot_.cpu_data()[j] - 2 * caffe_cpu_dot(dim, pos_data + dim * i, pos_data + dim * j);
				}
				neg_dist[ind] = pos_dot_.cpu_data()[i] + neg_dot_.cpu_data()[j] - 2 * caffe_cpu_dot(dim, pos_data + dim * i, neg_data + dim * j);
			}
		}		
		
		for (int i = 0; i < ncount; i++)
		{
			for (int j = 0; j < ncount; j++)
			{
				for (int k = 0; k < ncount; k++)
				{
					int ind = (i * ncount + j) * ncount + k;
					Dtype dloss = pos_dist_.cpu_data()[i * ncount + j] + margin - neg_dist_.cpu_data()[i * ncount + k];
					if (dloss > 0)
					{
						vec_diff_coeff[i * (2 * ncount) + j] -= 1;
						vec_diff_coeff[i * (2 * ncount) + ncount + k] += 1;
						vec_diff_coeff[j * (2 * ncount) + j] += 1;
						vec_diff_coeff[j * (2 * ncount) + i] -= 1;
						vec_diff_coeff[(ncount + k) * (2 * ncount) + i] += 1;
						vec_diff_coeff[(ncount + k) * (2 * ncount) + ncount + k] -= 1;
						loss += dloss;
					}
				}
			}
		}
	}

	top[0]->mutable_cpu_data()[0] = loss / 2 / (ncount * ncount * ncount) / batch_size;
}

template <typename Dtype>
void ClustersTripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	int dim = bottom[0]->shape(-1);
	int ncount = bottom[0]->shape(-2) / 2;
	int batch_size = (bottom[0]->num_axes() == 2) ? 1 : bottom[0]->num();

	if (propagate_down[0]) {
		//caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
		//for (int i = 0; i < bottom[0]->num(); i++)
		//{
		//	for (int j = 0; j < bottom[0]->num(); j++)
		//	{
		//		int c = vec_diff_coeff[i * (batch_size * 2) + j];
		//		caffe_axpy(dim, Dtype(c), bottom_data + j * dim, bottom_diff + i * dim);
		//	}
		//}

		for (int item = 0; item < batch_size; item++)
		{
			const Dtype* bottom_data = bottom[0]->cpu_data() + item * 2 * ncount * dim;
			const Dtype* vec_diff_coeff = vec_diff_coeff_.cpu_data() + item * ncount * ncount * 4;
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff() + item * 2 * ncount * dim;

			const Dtype alpha = top[0]->cpu_diff()[0] / (ncount * ncount * ncount) / batch_size;
			caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, ncount * 2, dim, ncount * 2, alpha,
				vec_diff_coeff, bottom_data, Dtype(0), bottom_diff);
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(ClustersTripletLossLayer);
#endif

INSTANTIATE_CLASS(ClustersTripletLossLayer);
REGISTER_LAYER_CLASS(ClustersTripletLoss);

}  // namespace caffe
