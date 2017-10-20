#include <algorithm>
#include <cfloat>
#include <vector>

// #include "thrust/device_vector.h"
// #include "device_atomic_functions.hpp"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/transformer_translation_layer.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void FixCoordinate(const int n, Dtype* in_out,
    Dtype min_value, Dtype max_value) {
    CUDA_KERNEL_LOOP(index, n) {
      in_out[index] = (in_out[index] < min_value && in_out[index] > min_value - 1e-4) ? min_value : in_out[index];
      in_out[index] = (in_out[index] > max_value && in_out[index] < (max_value + 1e-4)) ? max_value : in_out[index];
    }
  }

  template <typename Dtype>
  __global__ void TransformerTranslationForward(const int num, const int channels,
    const int spatial_dim, const int height, const int width,
    const Dtype* data, Dtype*  CoordinateSource_data,
    Dtype* transformed_data) {
    CUDA_KERNEL_LOOP(index, num * spatial_dim) {
      int n = index / spatial_dim;
      int s = index % spatial_dim;
      int h = s / width;
      int w = s % width;
      Dtype x = CoordinateSource_data[n * 2 * spatial_dim + h * width + w] * height / 2 + (Dtype)height / 2;
      Dtype y = CoordinateSource_data[n * 2 * spatial_dim + spatial_dim + h * width + w] * width / 2 + (Dtype)width / 2;
	  // x
	  Dtype x_temp = (Dtype)-1;
	  Dtype x0 = (x >= 0 && x <= height - 1) ? x:x_temp;
	  if (x0 == x_temp){
		  x0 = (x > 0) ? height - 1 : 0;
	  }
	  x = x0;
	  // y
	  Dtype y_temp = (Dtype)-1;
	  Dtype y0 = (y >= 0 && y <= width - 1) ? y:y_temp;
	  if (y0 == y_temp){
		  y0 = (y > 0) ? width - 1 : 0;
	  }
	  y = y0;
      // if (x >= 0 && x <= height - 1 && y >= 0 && y <= width - 1) {
        for (int c = 0; c < channels; c++) {
          for (int xx = floor(x); xx <= ceil(x); xx++) {
            for (int yy = floor(y); yy <= ceil(y); yy++) {
              transformed_data[(((n * channels + c) * height + h) * width) + w] += data[(((n * channels + c) * height + xx) * width) + yy] * (1 - abs(x - xx)) * (1 - abs(y - yy));
            }
          }
        }
      //}
    }
  }
  template <typename Dtype>
  __global__ void TransformerTranslationBackward(const int num, const int channels,
    const int spatial_dim, const int height, const int width,
    const Dtype* data, const Dtype* CoordinateSource_data, const Dtype* top_diff,
    Dtype* data_diff, Dtype* CoordinateSource_diff);

  template <>
  __global__ void TransformerTranslationBackward<float>(const int num, const int channels,
    const int spatial_dim, const int height, const int width,
    const float* data, const float* CoordinateSource_data, const float* top_diff,
    float* data_diff, float* CoordinateSource_diff) {
    CUDA_KERNEL_LOOP(index, num * spatial_dim) {
      int n = index / spatial_dim;
      int s = index % spatial_dim;
      int h = s / width;
      int w = s % width;
      float x = CoordinateSource_data[n * 2 * spatial_dim + h * width + w] * height / 2 + (float)height / 2;
      float y = CoordinateSource_data[n * 2 * spatial_dim + spatial_dim + h * width + w] * width / 2 + (float)width / 2;
	  // x
	  float x_temp = (float)-1;
	  float x0 = (x >= 0 && x <= height - 1) ? x : x_temp;
	  if (x0 == x_temp){
		  x0 = (x > 0) ? height - 1 : 0;
	  }
	  x = x0;
	  // y
	  float y_temp = (float)-1;
	  float y0 = (y >= 0 && y <= width - 1) ? y : y_temp;
	  if (y0 == y_temp){
		  y0 = (y > 0) ? width - 1 : 0;
	  }
	  y = y0;
      //if (x >= 0 && x <= height - 1 && y >= 0 && y <= width - 1) {
        for (int c = 0; c < channels; c++) {
          for (int xx = floor(x); xx <= ceil(x); xx++) {
            for (int yy = floor(y); yy <= ceil(y); yy++) {
              atomicAdd(&data_diff[(((n * channels + c) * height + xx) * width) + yy], top_diff[(((n * channels + c) * height + h) * width) + w] * (1 - abs(x - xx)) * (1 - abs(y - yy)));
              //printf("(%d,%d,%d,%d)(%f,%f)(%d,%d)(%f,%f)\n", n, c, h, w, x, y, xx, yy, data_diff[(((n * channels + c) * height + xx) * width) + yy], top_diff[(((n * channels + c) * height + h) * width) + w]);
              if ((xx - x)>float(0))
                CoordinateSource_diff[n * 2 * spatial_dim + h * width + w] += top_diff[(((n * channels + c) * height + h) * width) + w] * data[(((n * channels + c) * height + xx) * width) + yy] * (1 - abs(y - yy)) * (float)height / 2;
              else
                CoordinateSource_diff[n * 2 * spatial_dim + h * width + w] -= top_diff[(((n * channels + c) * height + h) * width) + w] * data[(((n * channels + c) * height + xx) * width) + yy] * (1 - abs(y - yy)) * (float)height / 2;
              if ((yy - y) > float(0))
                CoordinateSource_diff[n * 2 * spatial_dim + spatial_dim + h * width + w] += top_diff[(((n * channels + c) * height + h) * width) + w] * data[(((n * channels + c) * height + xx) * width) + yy] * (1 - abs(x - xx)) * (float)width / 2;
              else
                CoordinateSource_diff[n * 2 * spatial_dim + spatial_dim + h * width + w] -= top_diff[(((n * channels + c) * height + h) * width) + w] * data[(((n * channels + c) * height + xx) * width) + yy] * (1 - abs(x - xx)) * (float)width / 2;
            }
          }
        }
      //}
    }
  }

  __device__ double atomicAddTranslation(double* address, double val) {
    unsigned long long int* address_as_ull =
      (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val +
        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
  }

  template <>
  __global__ void TransformerTranslationBackward<double>(const int num, const int channels,
    const int spatial_dim, const int height, const int width,
    const double* data, const double* CoordinateSource_data, const double* top_diff,
    double* data_diff, double* CoordinateSource_diff) {
    CUDA_KERNEL_LOOP(index, num * spatial_dim) {
      int n = index / spatial_dim;
      int s = index % spatial_dim;
      int h = s / width;
      int w = s % width;
      double x = CoordinateSource_data[n * 2 * spatial_dim + h * width + w] * height / 2 + (double)height / 2;
      double y = CoordinateSource_data[n * 2 * spatial_dim + spatial_dim + h * width + w] * width / 2 + (double)width / 2;
	  // x
	  double x_temp = (double)-1;
	  double x0 = (x >= 0 && x <= height - 1) ? x:x_temp;
	  if (x0 == x_temp){
		  x0 = (x > 0) ? height - 1 : 0;
	  }
	  x = x0;
	  // y
	  double y_temp = (double)-1;
	  double y0 = (y >= 0 && y <= width - 1) ? y:y_temp;
	  if (y0 == y_temp){
		  y0 = (y > 0) ? width - 1 : 0;
	  }
	  y = y0;
      if (x >= 0 && x <= height - 1 && y >= 0 && y <= width - 1) {
        for (int c = 0; c < channels; c++) {
          for (int xx = floor(x); xx <= ceil(x); xx++) {
            for (int yy = floor(y); yy <= ceil(y); yy++) {
              //atomicAdd do not support double float. Please avoid using Net<double>.
				atomicAddTranslation(&data_diff[(((n * channels + c) * height + xx) * width) + yy], top_diff[(((n * channels + c) * height + h) * width) + w] * (1 - abs(x - xx)) * (1 - abs(y - yy)));
              //printf("(%d,%d,%d,%d)(%f,%f)(%d,%d)(%f,%f)\n", n, c, h, w, x, y, xx, yy, data_diff[(((n * channels + c) * height + xx) * width) + yy], top_diff[(((n * channels + c) * height + h) * width) + w]);
              if ((xx - x)>double(0))
                CoordinateSource_diff[n * 2 * spatial_dim + h * width + w] += top_diff[(((n * channels + c) * height + h) * width) + w] * data[(((n * channels + c) * height + xx) * width) + yy] * (1 - abs(y - yy)) * (double)height / 2;
              else
                CoordinateSource_diff[n * 2 * spatial_dim + h * width + w] -= top_diff[(((n * channels + c) * height + h) * width) + w] * data[(((n * channels + c) * height + xx) * width) + yy] * (1 - abs(y - yy)) * (double)height / 2;
              if ((yy - y) > double(0))
                CoordinateSource_diff[n * 2 * spatial_dim + spatial_dim + h * width + w] += top_diff[(((n * channels + c) * height + h) * width) + w] * data[(((n * channels + c) * height + xx) * width) + yy] * (1 - abs(x - xx)) * (double)width / 2;
              else
                CoordinateSource_diff[n * 2 * spatial_dim + spatial_dim + h * width + w] -= top_diff[(((n * channels + c) * height + h) * width) + w] * data[(((n * channels + c) * height + xx) * width) + yy] * (1 - abs(x - xx)) * (double)width / 2;
            }
          }
        }
      }
    }
  }

  template <typename Dtype>
  void TransformerTranslationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

	// 2, translation
	const Dtype* theta_data_old = bottom[1]->cpu_data();
	int theta_num = bottom[1]->shape(0);
	Dtype* theta_data_cpu = TempTheta.mutable_cpu_data();
	caffe_set<Dtype>(bottom[1]->count(), 0, theta_data_cpu);
	for (int i = 0; i < theta_num; ++i){
		//caffe_copy<Dtype>(1, theta_data_old+i*theta_num + 0, theta_data[i*theta_num + 2]);
		//caffe_copy<Dtype>(1, theta_data_old+i*theta_num + 1, theta_data[i*theta_num + 5]);
		theta_data_cpu[i*theta_num + 2]=theta_data_old[i*theta_num + 0];
		theta_data_cpu[i*theta_num + 5]=theta_data_old[i*theta_num + 1];	
	}
	const Dtype* theta_data = TempTheta.gpu_data();
	//

    const Dtype* CoordinateTarget_data = CoordinateTarget.gpu_data();
    Dtype*  CoordinateSource_data = CoordinateSource.mutable_gpu_data();
    int num = top[0]->shape(0);
    int channels = top[0]->shape(1);
    int spatial_dim = top[0]->shape(2) * top[0]->shape(3);
    caffe_gpu_set<Dtype>(top[0]->count(), 0, top_data);//why memset cause error?
    for (int n = 0; n < num; n++) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 2, spatial_dim, 3,
        Dtype(1), theta_data + n * 6, CoordinateTarget_data, Dtype(0), CoordinateSource_data + n * 2 * spatial_dim);
      FixCoordinate<Dtype> << <CAFFE_GET_BLOCKS(spatial_dim), CAFFE_CUDA_NUM_THREADS >> >(
        spatial_dim, CoordinateSource_data + n * 2 * spatial_dim, -1, 1 - 2 / (Dtype)bottom[0]->shape(2));//height = 10, then max = 9/5-1=0.8
      FixCoordinate<Dtype> << <CAFFE_GET_BLOCKS(spatial_dim), CAFFE_CUDA_NUM_THREADS >> >(
        spatial_dim, CoordinateSource_data + n * 2 * spatial_dim + spatial_dim, -1, 1 - 2 / (Dtype)bottom[0]->shape(3));
    }
	TransformerTranslationForward<Dtype> << <CAFFE_GET_BLOCKS(num * spatial_dim),
      CAFFE_CUDA_NUM_THREADS >> >(num, channels, spatial_dim,
      bottom[0]->shape(2), bottom[0]->shape(3),
      bottom_data, CoordinateSource_data, top_data);
  }

  template <typename Dtype>
  void TransformerTranslationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* top_data = top[0]->gpu_data();
    Dtype* data_diff = bottom[0]->mutable_gpu_diff();

	// 2, translation
	Dtype* theta_diff_old = bottom[1]->mutable_cpu_diff();
	Dtype* theta_diff = TempTheta.mutable_gpu_diff();
	caffe_gpu_set<Dtype>(bottom[1]->count(), 0, theta_diff);
	//

    int num = top[0]->shape(0);
    int channels = top[0]->shape(1);
    int spatial_dim = top[0]->shape(2) * top[0]->shape(3);
    const Dtype* CoordinateTarget_data = CoordinateTarget.gpu_data();
    const Dtype*  CoordinateSource_data = CoordinateSource.gpu_data();
    Dtype* CoordinateSource_diff = CoordinateSource.mutable_gpu_diff();

    caffe_gpu_set<Dtype>(bottom[0]->count(), 0, data_diff);
    caffe_gpu_set<Dtype>(CoordinateSource.count(), 0, CoordinateSource_diff);

	TransformerTranslationBackward<Dtype> << <CAFFE_GET_BLOCKS(num * spatial_dim),
      CAFFE_CUDA_NUM_THREADS >> >(num, channels, spatial_dim,
      bottom[0]->shape(2), bottom[0]->shape(3),
      bottom_data, CoordinateSource_data, top_diff,
      data_diff, CoordinateSource_diff);
    for (int n = 0; n < num; n++) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 2, 3, spatial_dim,
        Dtype(1), CoordinateSource_diff + n * 2 * spatial_dim, CoordinateTarget_data, Dtype(0), theta_diff + n * 6);
    }

	// 2, translation
	const Dtype* theta_diff_cpu = TempTheta.cpu_diff();
	for (int i = 0; i < num; ++i){
		//caffe_copy<Dtype>(1, theta_diff+i*num + 2, theta_diff_old+i*num + 0);
		//caffe_copy<Dtype>(1, theta_diff+i*num + 5, theta_diff_old+i*num + 1);
		theta_diff_old[i*num + 0] = theta_diff_cpu[i*num + 2];
		theta_diff_old[i*num + 1] = theta_diff_cpu[i*num + 5];
	}
	//

  }

  INSTANTIATE_LAYER_GPU_FUNCS(TransformerTranslationLayer);


}  // namespace caffe
