#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/key_point_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void KeyPointLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		KeyPointParameter key_point_param = this->layer_param_.key_point_param();
		height_ = key_point_param.region_height();
		width_ = key_point_param.region_width();
		data_height_ = key_point_param.data_height();
		data_width_ = key_point_param.data_width();
		key_scale_ = key_point_param.key_scale();
	}

	template <typename Dtype>
	void KeyPointLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
			<< "corresponding to (num, channels, height, width)";
		top[0]->Reshape({ bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width()});
	}

	template <typename Dtype>
	void KeyPointLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* point_data = bottom[1]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		caffe_copy(bottom[0]->count(), bottom_data, top_data);

		const int num = bottom[0]->num();
		const int channels = bottom[0]->channels();
		for (int n = 0; n < num; n++) {
			int point_count = 0;
			for (int i = 1; i < bottom[1]->shape(1); ++i){
				if (point_data[n*bottom[1]->shape(1) + i] != -100){
					point_count = point_count + 1;
				}
				else
					break;
			}
			const int num_point = point_count / 2;
			for (int i = 0; i < num_point; i++) {
				if (point_data[n*bottom[1]->shape(1) + i * 2 + 1] != -100)
				{
					Dtype center_x = (point_data[n * bottom[1]->shape(1) + i * 2 + 1] / data_width_ + 0.5)  * bottom[0]->width();
					Dtype center_y = (point_data[n * bottom[1]->shape(1) + i * 2 + 2] / data_height_ + 0.5) * bottom[0]->height();
					int x0 = floor(center_x - width_ / 2);
					int y0 = floor(center_y - height_ / 2);
					for (int c = 0; c < channels; c++) {
						for (int h = 0; h < height_; h++) {
							for (int w = 0; w < width_; w++) {
								if (y0 + h >= 0 && y0 + h <= bottom[0]->height() - 1
									&& x0 + w >= 0 && x0 + w <= bottom[0]->width() - 1) {
									top_data[top[0]->offset(n, c, y0 + h, x0 + w)] = bottom_data[bottom[0]->offset(n, c, y0 + h, x0 + w)] * key_scale_;
								}
							}
						}
					}//channel
				}
				else
					break;
			}//point
		}//num
	}

	template <typename Dtype>
	void KeyPointLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (!propagate_down[0]) {
			return;
		}
		const Dtype* top_diff = top[0]->cpu_diff();
		const Dtype* point_data = bottom[1]->cpu_data();
		Dtype* bottom_diff = top[0]->mutable_cpu_diff();
		caffe_copy(bottom[0]->count(), top_diff, bottom_diff);

		const int num = bottom[0]->num();
		const int channels = bottom[0]->channels();
		for (int n = 0; n < num; n++) {
			int point_count = 0;
			for (int i = 1; i < bottom[1]->shape(1); ++i){
				if (point_data[n*bottom[1]->shape(1) + i] != -100){
					point_count = point_count + 1;
				}
				else
					break;
			}
			const int num_point = point_count / 2;
			for (int i = 0; i < num_point; i++) {
				if (point_data[n*bottom[1]->shape(1) + i * 2 + 1] != -100)
				{
					Dtype center_x = (point_data[n * bottom[1]->shape(1) + i * 2 + 1] / data_width_ + 0.5)  * bottom[0]->width();
					Dtype center_y = (point_data[n * bottom[1]->shape(1) + i * 2 + 2] / data_height_ + 0.5) * bottom[0]->height();
					int x0 = floor(center_x - width_ / 2);
					int y0 = floor(center_y - height_ / 2);
					for (int c = 0; c < channels; c++) {
						for (int h = 0; h < height_; h++) {
							for (int w = 0; w < width_; w++) {
								if (y0 + h >= 0 && y0 + h <= bottom[0]->height() - 1
									&& x0 + w >= 0 && x0 + w <= bottom[0]->width() - 1) {
									bottom_diff[bottom[0]->offset(n, c, y0 + h, x0 + w)] = top_diff[top[0]->offset(n, c, y0 + h, x0 + w)] * key_scale_;
								}
							}
						}
					}//channel
				}
				else
					break;
			}//point
		}//num
	}


#ifdef CPU_ONLY
	STUB_GPU(KeyPointLayer);
#endif

	INSTANTIATE_CLASS(KeyPointLayer);
	REGISTER_LAYER_CLASS(KeyPoint);

}  // namespace caffe