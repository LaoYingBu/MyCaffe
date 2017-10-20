#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/clusters_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ClustersDataLayer<Dtype>::~ClustersDataLayer<Dtype>() {
	this->StopInternalThread();
}

template <typename Dtype>
void ClustersDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const int new_height = this->layer_param_.image_data_param().new_height();
	const int new_width = this->layer_param_.image_data_param().new_width();
	const bool is_color = this->layer_param_.image_data_param().is_color();
	const int batch_size = this->layer_param_.image_data_param().batch_size();
	string root_folder = this->layer_param_.image_data_param().root_folder();
	const int cluster_size = this->layer_param_.clusters_data_param().cluster_size();
	CHECK_GT(batch_size, 0) << "Positive batch size required";
	CHECK_GT(cluster_size, 0) << "Clusters data cluster_size must be > 0";
	CHECK_EQ(batch_size % cluster_size, 0) << "Clusters data batch_size must be devided by cluster_size";
	const int num_clusters = batch_size / cluster_size;

	CHECK((new_height == 0 && new_width == 0) ||
		(new_height > 0 && new_width > 0)) << "Current implementation requires "
		"new_height and new_width to be set at the same time.";
	// Read the file with filenames and labels
	const string& source = this->layer_param_.image_data_param().source();
	LOG(INFO) << "Opening file " << source;
	std::ifstream infile(source.c_str());
	string filename;
	int label;
	int count = 0;
	group_index_.clear();
	lines_.clear();
	while (infile >> filename >> label) {
		lines_.push_back(std::make_pair(filename, label));
		while (label >= group_index_.size())
		{
			group_index_.push_back(vector<int>());
		}
		group_index_[label].push_back(count++);
	}
	CHECK_GE(group_index_.size(), num_clusters) << "Clusters in a batch must not greater than the num of classes";

	// Do not shuffle nor skip

	//if (this->layer_param_.image_data_param().shuffle()) {
	//	// randomly shuffle data
	//	LOG(INFO) << "Shuffling data";
	//	const unsigned int prefetch_rng_seed = caffe_rng_rand();
	//	prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
	//	ShuffleImages();
	//}
	LOG(INFO) << "A total of " << lines_.size() << " images.";

	int lines_id = 0;
	// Check if we would need to randomly skip a few data points
	//if (this->layer_param_.image_data_param().rand_skip()) {
	//	unsigned int skip = caffe_rng_rand() %
	//		this->layer_param_.image_data_param().rand_skip();
	//	LOG(INFO) << "Skipping first " << skip << " data points.";
	//	CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
	//	lines_id = skip;
	//}
	// Read an image, and use it to initialize the top blob.
	cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id].first,
		new_height, new_width, is_color);
	CHECK(cv_img.data) << "Could not load " << lines_[lines_id].first;
	// Use data_transformer to infer the expected blob shape from a cv_image.
	vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
	this->transformed_data_.Reshape(top_shape);
	// Reshape prefetch_data and top[0] according to the batch_size.

	top_shape[0] = batch_size;
	for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
		this->prefetch_[i].data_.Reshape(top_shape);
	}
	top[0]->Reshape(top_shape);

	LOG(INFO) << "output data size: " << top[0]->num() << ","
		<< top[0]->channels() << "," << top[0]->height() << ","
		<< top[0]->width();
	// label
	vector<int> label_shape(1, batch_size);
	top[1]->Reshape(label_shape);
	for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
		this->prefetch_[i].label_.Reshape(label_shape);
	}
	// prefetch_rng_
	const unsigned int prefetch_rng_seed = caffe_rng_rand();
	prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
}

template <typename Dtype>
int ClustersDataLayer<Dtype>::Rand(int n) {
	CHECK(prefetch_rng_);
	CHECK_GT(n, 0);
	caffe::rng_t* prefetch_rng =
		static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	return ((*prefetch_rng)() % n);
}

//template <typename Dtype>
//void ClustersDataLayer<Dtype>::ShuffleImages() {
//	caffe::rng_t* prefetch_rng =
//		static_cast<caffe::rng_t*>(prefetch_rng_->generator());
//	shuffle(lines_.begin(), lines_.end(), prefetch_rng);
//}

// This function is called on prefetch thread
template <typename Dtype>
void ClustersDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
	CPUTimer batch_timer;
	batch_timer.Start();
	double read_time = 0;
	double trans_time = 0;
	CPUTimer timer;
	CHECK(batch->data_.count());
	CHECK(this->transformed_data_.count());
	ImageDataParameter image_data_param = this->layer_param_.image_data_param();
	const int batch_size = image_data_param.batch_size();
	const int new_height = image_data_param.new_height();
	const int new_width = image_data_param.new_width();
	const bool is_color = image_data_param.is_color();
	string root_folder = image_data_param.root_folder();
	const int cluster_size = this->layer_param_.clusters_data_param().cluster_size();
	const int num_clusters = batch_size / cluster_size;

	int lines_id = 0;
	// Reshape according to the first image of each batch
	// on single input batches allows for inputs of varying dimension.
	cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id].first,
		new_height, new_width, is_color);
	CHECK(cv_img.data) << "Could not load " << lines_[lines_id].first;
	// Use data_transformer to infer the expected blob shape from a cv_img.
	vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
	this->transformed_data_.Reshape(top_shape);
	// Reshape batch according to the batch_size.
	top_shape[0] = batch_size;
	batch->data_.Reshape(top_shape);

	Dtype* prefetch_data = batch->data_.mutable_cpu_data();
	Dtype* prefetch_label = batch->label_.mutable_cpu_data();

	// datum scales
	const int lines_size = lines_.size();
	int rand_size = lines_size;
	vector<int> cluster_label(num_clusters, -1);
	for (int i = 0; i < num_clusters; i++)
	{
		int cur_id = Rand(rand_size);
		int clabel = 0;
		int k = 0;
		while (clabel < group_index_.size())
		{
			if (clabel == cluster_label[k])
			{
				k++;
			}
			else
			{
				cur_id -= group_index_[clabel].size();
				if (cur_id < 0) break;
			}
			clabel++;
		}
		for (int j = i; j > k; j--)
		{
			cluster_label[j] = cluster_label[j - 1];
		}
		cluster_label[k] = clabel;
		rand_size = rand_size - group_index_[clabel].size();
	}
		
	for (int cluster_id = 0; cluster_id < num_clusters; cluster_id++)
	{
		rand_size = group_index_[cluster_label[cluster_id]].size();
		vector<int> rand_id(cluster_size, rand_size);
		//for (int cid = 0; cid < cluster_size; ++cid) rand_id[cid] = -1;
		for (int cid = 0; cid < cluster_size; ++cid)
		{
			if (rand_size > 0)
			{
				int rid = Rand(rand_size);
				int k = 0;
				while ((k < cid) && (rand_id[k] <= rid))
				{
					k++;
					rid++;
				}
				for (int i = cid; i > k; i--)
				{
					rand_id[i] = rand_id[i - 1];
				}
				rand_id[k] = rid;
				rand_size--;
			}
			else {
				rand_id[cid] = Rand(group_index_[cluster_label[cluster_id]].size());
			}
		}
		for (int cid = 0; cid < cluster_size; ++cid)
		{
			int item_id = cluster_id * cluster_size + cid;
			int cur_id = group_index_[cluster_label[cluster_id]][rand_id[cid]];
			// get a blob
			timer.Start();
	
			//CHECK_GT(lines_size, lines_id);
			cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[cur_id].first,
			new_height, new_width, is_color);
			CHECK(cv_img.data) << "Could not load " << lines_[cur_id].first;
			read_time += timer.MicroSeconds();
			timer.Start();
			//// Apply transformations (mirror, crop...) to the image
			int offset = batch->data_.offset(item_id);
			this->transformed_data_.set_cpu_data(prefetch_data + offset);
			this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
			trans_time += timer.MicroSeconds();

			prefetch_label[item_id] = lines_[cur_id].second;
		}
	}

	batch_timer.Stop();
	DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
	DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
	DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ClustersDataLayer);
REGISTER_LAYER_CLASS(ClustersData);

}  // namespace caffe
#endif  // USE_OPENCV
