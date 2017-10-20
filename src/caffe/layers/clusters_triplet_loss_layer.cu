//#include <algorithm>
//#include <vector>
//
//#include "caffe/layers/clusters_triplet_loss_layer.hpp"
//#include "caffe/util/math_functions.hpp"
//
//namespace caffe {
//
//	template <typename Dtype>
//	__global__ void ClustersTripletForward(const int nthreads, const int batch_size,
//		const int dim, const Dtype margin, const Dtype* pos_data,const Dtype* neg_data,
//		Dtype* vec_loss) {
//
//		CUDA_KERNEL_LOOP(index, nthreads) {
//			const int ind_a = index / (batch_size * batch_size);
//			const int ind_p = (index % (batch_size * batch_size)) / batch_size;
//			const int ind_n = index % batch_size;
//
//			Dtype dpa(0.0), dna(0.0), t;
//			for (int i = 0; i < dim; i++)
//			{
//				t = pos_data[ind_p * dim + i] - pos_data[ind_a * dim + i];
//				dpa += t * t;
//				t = neg_data[ind_n * dim + i] - pos_data[ind_a * dim + i];
//				dna += t * t;
//			}
//			vec_loss[index] = max(Dtype(0), dpa + margin - dna);
//		}
//	}
//
//	template <typename Dtype>
//	__global__ void ClustersTripletBackward(const int nthreads, const int batch_size,
//		const int dim, const Dtype* pos_data, const Dtype* neg_data, const Dtype* vec_loss,
//		Dtype* pos_diff, Dtype* neg_diff) {
//
//		CUDA_KERNEL_LOOP(index, nthreads) {
//			const int item = index / dim;
//			const int k = index % dim;
//
//			Dtype diff;
//			for (int i = 0; i < batch_size; i++)
//			{
//				for (int j = 0; j < batch_size; j++)
//				{
//					if (vec_loss[(item * batch_size + i) * batch_size + j] > 0)
//					{
//						pos_diff[item * dim + k] += neg_data[j * dim + k] - pos_data[i * dim + k];
//					}
//
//					if (vec_loss[(i * batch_size + item) * batch_size + j] > 0)
//					{
//						pos_diff[item * dim + k] += pos_data[item * dim + k] - pos_data[i * dim + k];
//					}
//
//					if (vec_loss[(i * batch_size + j) * batch_size + item] > 0)
//					{
//						neg_diff[item * dim + k] += pos_data[i * dim + k] - neg_data[item * dim + k];
//					}
//				}
//			}
//		}
//	}
//	
//	template <typename Dtype>
//	void ClustersTripletLossLayer<Dtype>::Forward_gpu(
//		const vector<Blob<Dtype>*>& bottom,
//		const vector<Blob<Dtype>*>& top) {
//
//		//ClustersTripletLossLayer<Dtype>::Forward_cpu(bottom, top);
//
//		Dtype margin = this->layer_param_.contrastive_loss_param().margin();
//		int batch_size = bottom[0]->num() / 2;
//		int dim = bottom[0]->channels();
//		const Dtype* pos_data = bottom[0]->gpu_data();
//		const Dtype* neg_data = bottom[0]->gpu_data() + batch_size * dim;
//		Dtype* vec_loss = vec_loss_.mutable_gpu_data();
//		Dtype nthreads = batch_size * batch_size * batch_size;
//		caffe_gpu_set(vec_loss_.count(), Dtype(0), vec_loss);
//		ClustersTripletForward<Dtype> << <CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS >> >(
//			nthreads, batch_size, dim, margin, pos_data, neg_data, vec_loss);
//
//		Dtype loss(0.0);
//		for (int i = 0; i < vec_loss_.count(); i++)
//		{
//			loss += vec_loss_.cpu_data()[i];
//		}
//		top[0]->mutable_cpu_data()[0] = loss / batch_size / 2;
//	}
//
//	template <typename Dtype>
//	void ClustersTripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
//		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//
//		//ClustersTripletLossLayer<Dtype>::Backward_cpu(top, propagate_down, bottom);
//		
//		int batch_size = bottom[0]->num() / 2;
//		int dim = bottom[0]->channels();
//		const Dtype* pos_data = bottom[0]->gpu_data();
//		const Dtype* neg_data = bottom[0]->gpu_data() + batch_size * dim;
//		const Dtype* vec_loss = vec_loss_.gpu_data();
//		Dtype* pos_diff = bottom[0]->mutable_gpu_diff();
//		Dtype* neg_diff = bottom[0]->mutable_gpu_diff() + batch_size * dim;
//		Dtype nthreads = batch_size * dim;
//
//		if (propagate_down[0]) {
//			caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
//			ClustersTripletBackward<Dtype> << <CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS >> >(
//				nthreads, batch_size, dim, pos_data, neg_data, vec_loss, pos_diff, neg_diff);
//			const Dtype alpha = top[0]->cpu_diff()[0] / batch_size;
//			caffe_gpu_scal(bottom[0]->count(), alpha, bottom[0]->mutable_gpu_diff());
//		}
//
//	}
//
//INSTANTIATE_LAYER_GPU_FUNCS(ClustersTripletLossLayer);
//
//}  // namespace caffe
