#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/transformer_translation_layer.hpp"

namespace caffe {

  template <typename Dtype>
  void TransformerTranslationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	vector<int> top_shape(bottom[0]->shape());
	if(this->layer_param().transformer_param().top_height()){
      top_shape[2] = this->layer_param().transformer_param().top_height();
	  top_shape[3] = this->layer_param().transformer_param().top_width();
	}
	else{
	  top_shape[2] = bottom[0]->shape(2);
	  top_shape[3] = bottom[0]->shape(3);
	}
    top[0]->Reshape(top_shape);
	
	int num_theta = this->layer_param().transformer_param().num_theta();
    CHECK(bottom[1]->shape(1) == num_theta) << "Theta's dimension must be 6.";

    int shape1[3] = { 1, 3, top[0]->shape(2) * top[0]->shape(3) };
    vector<int> CoordinateTarget_shape(shape1, shape1 + 3);

    CoordinateTarget.Reshape(CoordinateTarget_shape);
    Dtype* CoordinateTarget_data = CoordinateTarget.mutable_cpu_data();
    for (int i = 0; i < top[0]->shape(2); i++) {
      for (int j = 0; j < top[0]->shape(3); j++) {
        CoordinateTarget_data[i * top[0]->shape(2) + j] = (Dtype)i / (Dtype)top[0]->shape(2) * 2 - 1;
        CoordinateTarget_data[i * top[0]->shape(2) + j + CoordinateTarget.shape(2)] = (Dtype)j / (Dtype)top[0]->shape(3) * 2 - 1;
        CoordinateTarget_data[i * top[0]->shape(2) + j + CoordinateTarget.shape(2) * 2] = 1;
      }
    }
    int shape2[4] = { top[0]->shape(0), 2, top[0]->shape(2), top[0]->shape(3) };
    vector<int> CoordinateSource_shape(shape2, shape2 + 4);
    CoordinateSource.Reshape(CoordinateSource_shape);
	// additional parameter 
	vector<int> theta_shape(bottom[1]->shape());
	theta_shape[1] = 6;
	TempTheta.Reshape(theta_shape);
  }

  template <typename Dtype>
  void TransformerTranslationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();


	// 2, translation
	const Dtype* theta_data_old = bottom[1]->cpu_data();
	int theta_num = bottom[1]->num();
	Dtype* theta_data = TempTheta.mutable_cpu_data();
	caffe_set<Dtype>(bottom[1]->count(), 0, theta_data);
	for (int i = 0; i < theta_num; ++i){
		//caffe_copy<Dtype>(1, theta_data_old+i*theta_num + 0, theta_data+i*theta_num + 2);
		//caffe_copy<Dtype>(1, theta_data_old+i*theta_num + 1, theta_data+i*theta_num + 5);
		theta_data[i*theta_num + 2] = theta_data_old[i*theta_num + 0];
		theta_data[i*theta_num + 5] = theta_data_old[i*theta_num + 1];
	}
	//

    const Dtype* CoordinateTarget_data = CoordinateTarget.cpu_data();
    Dtype*  CoordinateSource_data = CoordinateSource.mutable_cpu_data();
    int num = top[0]->shape(0);
    int spatial_dim = top[0]->shape(2) * top[0]->shape(3);

    caffe_set<Dtype>(top[0]->count(), 0, top_data);
    for (int n = 0; n < num; n++) {
		/*
		LOG(INFO) << "theta:" << theta_data[n * 6 + 0] << " " << theta_data[n * 6 + 1] << " " << theta_data[n * 6 + 2] << std::endl
			<< theta_data[n * 6 + 3] << " " << theta_data[n * 6 + 4] << " " << theta_data[n * 6 + 5];
	  */
	  /*
	  LOG(INFO) << CoordinateTarget_data[0] << " " << CoordinateTarget_data[1] << " " << CoordinateTarget_data[2] << std::endl
		  << CoordinateTarget_data[spatial_dim + 0] << " " << CoordinateTarget_data[spatial_dim + 1] << " " << CoordinateTarget_data[spatial_dim + 2] << std::endl
		  << CoordinateTarget_data[2 * spatial_dim + 0] << " " << CoordinateTarget_data[2 * spatial_dim + 1] << " " << CoordinateTarget_data[2 * spatial_dim + 2] << std::endl;
	  */
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 2, CoordinateTarget.shape(2), 3,
        Dtype(1), theta_data + n * 6, CoordinateTarget_data, Dtype(0), CoordinateSource_data + n * 2 * spatial_dim);
	  /*
      CoordinateSource_data += n * 2 * spatial_dim;
      LOG(INFO) << CoordinateSource_data[0] << " " << CoordinateSource_data[1] << " " << CoordinateSource_data[2] << std::endl
      << CoordinateSource_data[3] << " " << CoordinateSource_data[4] << " " << CoordinateSource_data[5] << std::endl
      << CoordinateSource_data[6] << " " << CoordinateSource_data[7] << " " << CoordinateSource_data[8] << std::endl
      << CoordinateSource_data[9] << " " << CoordinateSource_data[10] << " " << CoordinateSource_data[11] << std::endl
      << CoordinateSource_data[12] << " " << CoordinateSource_data[13] << " " << CoordinateSource_data[14] << std::endl
      << CoordinateSource_data[15] << " " << CoordinateSource_data[16] << " " << CoordinateSource_data[17];
      CoordinateSource_data -= n * 2 * spatial_dim;
	  */
      for (int i = 0; i < top[0]->shape(2); i++) {
        for (int j = 0; j < top[0]->shape(3); j++) {
          Dtype x = CoordinateSource_data[CoordinateSource.offset(n, 0, i, j)] * bottom[0]->shape(2) / 2 + (Dtype)bottom[0]->shape(2) / 2;
          Dtype y = CoordinateSource_data[CoordinateSource.offset(n, 1, i, j)] * bottom[0]->shape(3) / 2 + (Dtype)bottom[0]->shape(3) / 2;
           //LOG(INFO) << x << " " << y;
		  // x
		  Dtype x_temp = (Dtype)-1;
		  Dtype x0 = (x >= 0 && x <= bottom[0]->shape(2) - 1) ? x:x_temp;
		  if (x0 == x_temp){
			  x0 = (x > 0) ? bottom[0]->shape(2) - 1 : 0;
		  }
		  x = x0;
		  // y
		  Dtype y_temp = (Dtype)-1;
		  Dtype y0 = (y >= 0 && y <= bottom[0]->shape(3) - 1) ? y:y_temp;
		  if (y0 == y_temp){
			  y0 = (y > 0) ? bottom[0]->shape(3) - 1 : 0;
		  }
		  y = y0;
         //  if (x >= 0 && x <= CoordinateSource.shape(2) - 1 && y >= 0 && y <= CoordinateSource.shape(3) - 1) {
            for (int c = 0; c < top[0]->shape(1); c++) {
              for (int xx = floor(x); xx <= ceil(x); xx++) {
                for (int yy = floor(y); yy <= ceil(y); yy++) {
                  top_data[top[0]->offset(n, c, i, j)] += bottom[0]->data_at(n, c, xx, yy) * (1 - abs(x - xx)) * (1 - abs(y - yy));
                  //LOG(INFO) <<"("<< n << " " << c << " " << i << " " << j << ")("<<x<<","<<y<<")("<<xx<<","<<yy<<") " << top_data[top[0]->offset(n, c, i, j)];
                }
              }
            }
         //}
        }
      }
    }
  }

  template <typename Dtype>
  void TransformerTranslationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* top_data = top[0]->cpu_data();
    Dtype* data_diff = bottom[0]->mutable_cpu_diff();

	// 2, translation
	Dtype* theta_diff_old = bottom[1]->mutable_cpu_diff();
	Dtype* theta_diff = TempTheta.mutable_cpu_diff();
	caffe_set<Dtype>(bottom[1]->count(), 0, theta_diff);
	//

    int num = top[0]->shape(0);
    int spatial_dim = top[0]->shape(2) * top[0]->shape(3);
    const Dtype* CoordinateTarget_data = CoordinateTarget.cpu_data();
    const Dtype*  CoordinateSource_data = CoordinateSource.cpu_data();
    Dtype* CoordinateSource_diff = CoordinateSource.mutable_cpu_diff();

    caffe_set<Dtype>(bottom[0]->count(), 0, data_diff);
    caffe_set<Dtype>(CoordinateSource.count(), 0, CoordinateSource_diff);
    for (int n = 0; n < num; n++) {
      for (int i = 0; i < top[0]->shape(2); i++) {
        for (int j = 0; j < top[0]->shape(3); j++) {
          Dtype x = CoordinateSource_data[CoordinateSource.offset(n, 0, i, j)] * bottom[0]->shape(2) / 2 + (Dtype)bottom[0]->shape(2) / 2;
          Dtype y = CoordinateSource_data[CoordinateSource.offset(n, 1, i, j)] * bottom[0]->shape(3) / 2 + (Dtype)bottom[0]->shape(3) / 2;
		  // x
		  Dtype x_temp = (Dtype)-1;
		  Dtype x0 = (x >= 0 && x <= bottom[0]->shape(2) - 1) ? x:x_temp;
		  if (x0 == x_temp){
			  x0 = (x > 0) ? bottom[0]->shape(2) - 1 : 0;
		  }
		  x = x0;
		  // y
		  Dtype y_temp = (Dtype)-1;
		  Dtype y0 = (y >= 0 && y <= bottom[0]->shape(3) - 1) ? y:y_temp;
		  if (y0 == y_temp){
			  y0 = (y > 0) ? bottom[0]->shape(3) - 1 : 0;
		  }
		  y = y0;
          // if (x >= 0 && x <= CoordinateSource.shape(2) - 1 && y >= 0 && y <= CoordinateSource.shape(3) - 1) {
            for (int c = 0; c < top[0]->shape(1); c++) {
              for (int xx = floor(x); xx <= ceil(x); xx++) {
                for (int yy = floor(y); yy <= ceil(y); yy++) {
                  data_diff[bottom[0]->offset(n, c, xx, yy)] += top_diff[top[0]->offset(n, c, i, j)] * (1 - abs(x - xx)) * (1 - abs(y - yy));
                  //LOG(INFO) << n << " " << c << " " << i << " " << j << " " << data_diff[bottom[0]->offset(n, c, xx, yy)];
                  CoordinateSource_diff[CoordinateSource.offset(n, 0, i, j)] += top_diff[top[0]->offset(n, c, i, j)] * bottom[0]->data_at(n, c, xx, yy) * caffe_sign<Dtype>(xx - x) * (1 - abs(y - yy)) * (Dtype)top[0]->shape(2) / 2;
                  CoordinateSource_diff[CoordinateSource.offset(n, 1, i, j)] += top_diff[top[0]->offset(n, c, i, j)] * bottom[0]->data_at(n, c, xx, yy) * (1 - abs(x - xx)) * caffe_sign<Dtype>(yy - y) * (Dtype)top[0]->shape(3) / 2;
				  //LOG(INFO) << n << " " << 0 << " " << i << " " << j << " " << CoordinateSource_diff[CoordinateSource.offset(n, 0, i, j)];
				  //LOG(INFO) << n << " " << 1 << " " << i << " " << j << " " << CoordinateSource_diff[CoordinateSource.offset(n, 1, i, j)];
                }
              }
            }
         //}
        }
      }
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 2, 3, CoordinateTarget.shape(2),
        Dtype(1), CoordinateSource_diff + n * 2 * spatial_dim, CoordinateTarget_data, Dtype(0), theta_diff + n * 6);
    }

	// 2, translation
	for (int i = 0; i < num; ++i){
		theta_diff_old[i*num + 0] = theta_diff[i*num + 2];
		theta_diff_old[i*num + 1] = theta_diff[i*num + 5];
		LOG(INFO) << "theta_diff:" << theta_diff[i*num + 0] << " " << theta_diff[i*num + 1] << " " << theta_diff[i*num + 2] << std::endl
			<< theta_diff[i*num + 3] << " " << theta_diff[i*num + 4] << " " << theta_diff[i*num + 5];
	}
	//


  }


#ifdef CPU_ONLY
  STUB_GPU(TransformerTranslationLayer);
#endif

  INSTANTIATE_CLASS(TransformerTranslationLayer);
  REGISTER_LAYER_CLASS(TransformerTranslation);

}  // namespace caffe
