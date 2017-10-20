#include <vector>

#include "caffe/layers/rotate_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RotateConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK(this->layer_param().has_convolution_param());
	CHECK_EQ(bottom[0]->height(), bottom[0]->width());
	flip_ = (this->layer_param().has_transform_param()) ? (this->layer_param().transform_param().mirror) : true;
	exnum_ = flip_ ? 8 : 4;

	convolution_bottom_vec_.clear();
	convolution_top_vec_.clear();
	convolution_inputs_.resize(exnum_);
	convolution_outputs_.resize(exnum_);	
	for (int i = 0; i < exnum_; i++)
	{
		convolution_inputs_[i].reset(new Blob(bottom[0]->shape));
		convolution_bottom_vec_.push_back(convolution_inputs_[i].get());
		convolution_outputs_[i].reset(new Blob());
		convolution_top_vec_.push_back(convolution_outputs_[i].get());
	}
	convolution_layer_->mutable_layer_param() = this->layer_param();
	convolution_layer_->SetUp(convolution_bottom_vec_, convolution_top_vec_);
	this->blobs_[0]->ShareData(convolution_layer_->blobs(0));
	this->blobs_[1]->ShareData(convolution_layer_->blobs(1));

	eltwise_bottom_vec_.clear();
	eltwise_top_vec_.clear();
	eltwise_inputs_.resize(exnum_);
	for (int i = 0; i < exnum_; i++)
	{
		eltwise_inputs_[i].reset(new Blob(convolution_outputs_[i]->shape));
		eltwise_bottom_vec_.push_back(eltwise_inputs_[i].get());
	}
	eltwise_top_vec_.push_back(top[0]);
	eltwise_layer_->mutable_layer_param().mutable_eltwise_param().set_operation(EltwiseParameter_EltwiseOp_MAX);
	eltwise_layer_.SetUp(eltwise_bottom_vec_, eltwise_top_vec_);
}

template <typename Dtype>
void RotateConvLayer<Dtype>::RotateIndex(const int height, const int width, const int orient, 
	int& idx0, int& hstep, int& wstep)
{
	switch (orient)
	{
	case 0:
		idx0 = 0;
		hstep = width;
		wstep = 1;
		break;
	case 1:
		idx0 = width * (height - 1);
		hstep = 1;
		wstep = -width;
		break;
	case 2:
		idx0 = width * height - 1;
		hstep = -width;
		wstep = -1;
		break;
	case 3:
		idx0 = width - 1;
		hstep = -1;
		wstep = width;
		break;
	case 4:
		idx0 = width - 1;
		hstep = width;
		wstep = -1;
		break;
	case 5:
		idx0 = width * height - 1;
		hstep = -1;
		wstep = -width;
		break;
	case 6:
		idx0 = width * (height - 1);
		hstep = -width;
		wstep = 1;
		break;
	case 7:
		idx0 = 0;
		hstep = 1;
		wstep = width;
		break;
	default:
		LOG(ERROR) << "Unknown orientation";
	}
}

template <typename Dtype>
void RotateConvLayer<Dtype>::RotateMap(const Dtype* src, Dtype* dst, const int outer_dim,
	const int height,	const int width, const int orient, bool inverse)
{
	int idx0, hstep, wstep;
	RotateIndex(height, width, orient, idx0, hstep, wstep);

	for (int i = 0; i < outer_dim; i++)
	{
		int idx = idx0;
		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width; w++)
			{
				if (inverse)
				{
					dst[h * width + w] += src[idx];
				}
				else 
				{
					dst[idx] += src[h * width + w];
				}
				idx += wstep;
			}
			idx += hstep - wstep * width;
		}
		src += i * height * width;
		dst += i * height * width;
	}
}

template <typename Dtype>
void RotateConvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	int outer_dim = bottom[0]->num() * bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();

	for (int orient = 0; orient < exnum_; orient++)
	{
		caffe_set(convolution_inputs_[orient]->count(), Dtype(0), convolution_inputs_[orient]->mutable_cpu_data());
		caffe_set(eltwise_inputs_[orient]->count(), Dtype(0), eltwise_inputs_[orient]->mutable_cpu_data());
	}

	for (int orient = 0; orient < exnum_; orient++)
	{
		RotateMap(bottom[0]->cpu_data(), convolution_inputs_[orient]->mutable_cpu_data(),
			outer_dim, height, width, orient, false);
	}
	convolution_layer_->forward_cpu(convolution_bottom_vec_, convolution_top_vec_);

	height = top[0]->height();
	width = top[0]->width();
	for (int orient = 0; orient < exnum_; orient++)
	{		
		RotateMap(convolution_outputs_[orient]->cpu_data(), eltwise_inputs_[orient]->mutable_cpu_data(),
			outer_dim, height, width, orient, true);
	}
	eltwise_layer_->forward_cpu(eltwise_bottom_vec_, eltwise_top_vec_);
}

template <typename Dtype>
void RotateConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	int outer_dim = bottom[0]->num() * bottom[0]->channels();
	int height = top[0]->height();
	int width = top[0]->width();

	for (int orient = 0; orient < exnum_; orient++) {
		caffe_set(eltwise_outputs_[orient]->count(), Dtype(0), eltwise_outputs_[orient]->mutable_cpu_diff());
		caffe_set(convolution_outputs_[orient]->count(), Dtype(0), convolution_outputs_[orient]->mutable_cpu_diff());
	}
	caffe_set(bottom[0].count(), Dtype(0), bottom[0].mutable_cpu_diff());

	eltwise_layer_->backward_cpu(eltwise_top_vec_, propagate_down, eltwise_bottom_vec_);
	for (int orient = 0; orient < exnum_; orient++) {
		RotateMap(eltwise_inputs_[orient]->cpu_diff(), convolution_outputs_[orient]->mutable_cpu_diff(),
			outer_dim, height, width, orient, false);
	}
	
	convolution_layer_->backward_cpu(convolution_top_vec_, propagate_down, convolution_bottom_vec_);
	height = bottom[0]->height();
	width = bottom[0]->width();
	for (int orient = 0; orient < exnum_; orient++) {
		RotateMap(convolution_inputs_[orient].cpu_diff(), bottom[0]->mutable_cpu_diff(),
			outer_dim, height, width, orient, true);
	}
}

#ifdef CPU_ONLY
STUB_GPU(RotateConvLayer);
#endif

INSTANTIATE_CLASS(RotateConvLayer);
REGISTER_LAYER_CLASS(RotateConv);

}  // namespace caffe
