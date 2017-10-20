#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/channel_transpose_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ChannelTransposeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  ChannelTransposeLayerTest()
    : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
      blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ChannelTransposeLayerTest() { delete blob_bottom_; delete blob_top_; }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ChannelTransposeLayerTest, TestDtypesAndDevices);

TYPED_TEST(ChannelTransposeLayerTest, TestOutputSizes) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  ChannelTransposeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 6 * 5);
  EXPECT_EQ(this->blob_top_->height(), 3);
}

TYPED_TEST(ChannelTransposeLayerTest, TestValues) {
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	ChannelTransposeLayer<Dtype> layer(layer_param);
	layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	for (int c = 0; c < 3 * 6 * 5; ++c) {
		EXPECT_EQ(this->blob_top_->data_at(0, c % (6 * 5), c / (6 * 5), 0),
			this->blob_bottom_->data_at(0, c / (6 * 5), (c / 5) % 6, c % 5));
		EXPECT_EQ(this->blob_top_->data_at(1, c % (6 * 5), c / (6 * 5), 0),
			this->blob_bottom_->data_at(1, c / (6 * 5), (c / 5) % 6, c % 5));
	}
}

//TYPED_TEST(ChannelTransposeLayerTest, TestGradient) {
//  typedef typename TypeParam::Dtype Dtype;
//  LayerParameter layer_param;
//  ChannelTransposeLayer<Dtype> layer(layer_param);
//  GradientChecker<Dtype> checker(1e-2, 1e-2);
//  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
//      this->blob_top_vec_);
//}

}  // namespace caffe
