#ifndef CAFFE_CONTRASTIVE_DATA_LAYER_HPP_
#define CAFFE_CONTRASTIVE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	/**
	* @brief Provides contrastive datas to the Net from image files.
	*
	* tops: data 0 + data 1
	* 		  label 0 + label 1
	*	      simlabel
	*
	* TODO(dox): thorough documentation for Forward and proto params.
	*/
	template <typename Dtype>
	class ContrastiveDataLayer : public BasePrefetchingDataLayer<Dtype> {
	public:
		explicit ContrastiveDataLayer(const LayerParameter& param)
			: BasePrefetchingDataLayer<Dtype>(param) {}
		virtual ~ContrastiveDataLayer();
		virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ContrastiveData"; }
		virtual inline int ExactNumBottomBlobs() const { return 0; }
		virtual inline int ExactNumTopBlobs() const { return 3; }
		virtual	inline int Rand(int n);
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	protected:
		shared_ptr<Caffe::RNG> prefetch_rng_;
		//virtual void ShuffleImages();
		virtual void load_batch(Batch<Dtype>* batch);

		vector<std::pair<std::string, int> > lines_;
		vector<vector<int> > group_index_;
		//int lines_id_;
	};


}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
