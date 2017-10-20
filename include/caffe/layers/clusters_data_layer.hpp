#ifndef CAFFE_CLUSTERS_DATA_LAYER_HPP_
#define CAFFE_CLUSTERS_DATA_LAYER_HPP_

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
	* @brief Provides coupled clusters datas to the Net from image files.
	*
	* data 0 is sampled from the same identity
	* tops: data 0 + data 1
	* 		  label 0 + label 1
	*
	* TODO(dox): thorough documentation for Forward and proto params.
	*/
	template <typename Dtype>
	class ClustersDataLayer : public BasePrefetchingDataLayer<Dtype> {
	public:
		explicit ClustersDataLayer(const LayerParameter& param)
			: BasePrefetchingDataLayer<Dtype>(param) {}
		virtual ~ClustersDataLayer();
		virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ClustersData"; }
		virtual inline int ExactNumtopBlobs() const { return 2; }
		virtual	inline int Rand(int n);

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
