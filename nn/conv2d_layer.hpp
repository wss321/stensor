/**
* Copyright 2021 wss
* Created by wss on 11月,26, 2021
*/
#ifndef STENSOR_NN_CONV2D_LAYER_HPP_
#define STENSOR_NN_CONV2D_LAYER_HPP_
#include <cudnn.h>
#include "module.hpp"

namespace stensor {

namespace nn {
#define CUDNN_STREAMS_PER_GROUP 3

#define CUDNN_CHECK(condition) \
  do { \
    cudnnStatus_t status = condition; \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << " "\
      << cudnnGetErrorString(status); \
  } while (0)

template<typename Dtype>
class dataType;
template<>
class dataType<float> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
  static float oneval, zeroval;
  static const void *one, *zero;
};
template<>
class dataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
  static double oneval, zeroval;
  static const void *one, *zero;
};

class Conv2d : public Module {
 public:
  explicit Conv2d(const std::string &name,
                  int in_channels, int out_channels,
                  int kernel_h, int kernel_w,
                  int stride_h, int stride_w,
                  int padding_h, int padding_w,
                  int dilation_h, int dilation_w,
                  int group = 1,
                  int device = 0,
                  bool bias = true);
  ~Conv2d() override;
  TensorVec forward(TensorVec &inputs);
  void backward();
 private:
  void InitCuDnn();
  inline std::vector<int> calc_out_shape(const std::vector<int> &in_shape) {
    // B C H W
    CHECK_EQ(in_shape.size(), 4) << "Expected shape has 4 axes:[B, C, H, W]";
    int ow = (in_shape[3] - kernel_w_ + 2 * padding_w_) / stride_w_ + 1;
    int oh = (in_shape[2] - kernel_h_ + 2 * padding_h_) / stride_h_ + 1;
    return {in_shape[0], out_channels_, oh, ow};
  }
  void setUpConvAlg(const std::vector<int> &in_shape);
 private:
  int in_channels_, out_channels_;
  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int padding_h_, padding_w_;
  int dilation_h_, dilation_w_;
  int group_;
  int device_;
  bool bias_;

  SharedTensor kernel_;
  SharedTensor b_;
  SharedTensor result_;

  bool handles_setup_;
  cudnnHandle_t handle_;
  cudaStream_t stream_;
  // algorithms for forward and backwards convolutions
  cudnnConvolutionFwdAlgo_t fwd_algo_;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;

  cudnnTensorDescriptor_t input_desc_, output_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
//  int bottom_offset_, top_offset_, bias_offset_;

  size_t workspace_fwd_sizes_;
  size_t workspace_bwd_data_sizes_;
  size_t workspace_bwd_filter_sizes_;
  size_t workspaceSizeInBytes;  // size of underlying storage
  void *workspaceData;  // underlying storage
  void *workspace;  // aliases into workspaceData
};
}//namespace nn

}//namespace stensor
#endif //STENSOR_NN_CONV2D_LAYER_HPP_
