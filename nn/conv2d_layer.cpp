/**
* Copyright 2021 wss
* Created by wss on 11月,26, 2021
*/
#include "conv2d_layer.hpp"
#include "math/math_base_cuda.hpp"
#include "math/math_base_cpu.hpp"
#include "core/math_tesnsor.hpp"
namespace stensor {
namespace nn {

template<typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t *desc,
                            int n, int c, int h, int w,
                            int stride_n, int stride_c, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(*desc, dataType<Dtype>::type,
                                           n, c, h, w, stride_n, stride_c, stride_h, stride_w));
}

template<typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t *desc,
                            int n, int c, int h, int w) {
  const int stride_w = 1;
  const int stride_h = w * stride_w;
  const int stride_c = h * stride_h;
  const int stride_n = c * stride_c;
  setTensor4dDesc<Dtype>(desc, n, c, h, w,
                         stride_n, stride_c, stride_h, stride_w);
}

void Conv2d::InitCuDnn() {
//  stream_ =
  cudnnCreate(&handle_);
  cudaStreamCreate(&stream_);
  cudnnSetStream(handle_, stream_);
//  handle_ = Config::cudnn_handle();
  cudnnCreateFilterDescriptor(&filter_desc_);
  cudnnSetFilter4dDescriptor(filter_desc_,
                             CUDNN_DATA_FLOAT,
                             CUDNN_TENSOR_NCHW,
                             kernel_->shape(0),
                             kernel_->shape(1),
                             kernel_->shape(2),
                             kernel_->shape(3));
  if (bias_) {
    cudnnCreateTensorDescriptor(&bias_desc_);
    setTensor4dDesc<float>(&bias_desc_, 1, b_->size(), 1, 1);
  }

  cudnnCreateConvolutionDescriptor(&conv_desc_);
  cudnnSetConvolution2dDescriptor(conv_desc_,
                                  padding_h_, padding_w_, // zero-padding
                                  stride_h_, stride_w_, // stride
                                  dilation_h_, dilation_w_,
                                  CUDNN_CROSS_CORRELATION,
                                  dataType<float>::type);
  cudnnSetConvolutionGroupCount(conv_desc_, group_);

  workspace_fwd_sizes_ = 0;
  workspace_bwd_filter_sizes_ = 0;
  workspace_bwd_data_sizes_ = 0;
//  fwd_algo_ = (cudnnConvolutionFwdAlgo_t)0;
//  bwd_filter_algo_ = (cudnnConvolutionBwdFilterAlgo_t)0;
//  bwd_data_algo_ = (cudnnConvolutionBwdDataAlgo_t)0;
  fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
  bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  workspace = nullptr;
  workspaceSizeInBytes = 0;
  workspaceData = nullptr;
}

Conv2d::Conv2d(const std::string &name,
               int in_channels, int out_channels,
               int kernel_h, int kernel_w,
               int stride_h, int stride_w,
               int padding_h, int padding_w,
               int dilation_h, int dilation_w,
               int group,
               int device,
               bool bias) :
    in_channels_(in_channels), out_channels_(out_channels),
    kernel_h_(kernel_h), kernel_w_(kernel_w),
    stride_h_(stride_h), stride_w_(stride_w),
    padding_h_(padding_h), padding_w_(padding_w),
    dilation_h_(dilation_h), dilation_w_(dilation_w),
    group_(group), device_(device),
    bias_(bias) {
  this->name_ = name;
  this->type_ = "Conv2d";
  this->state_ = device_ > -1 ? GPU : CPU;
  if (bias_)
    this->parameters_.resize(2);
  else this->parameters_.resize(1);
  CHECK(in_channels % group == 0) << "wrong group value";
  Tensor *K = stensor::random({out_channels_, in_channels_, kernel_h_, kernel_w_}, -1, 1, device, true);
  K->set_name(name_ + "(" + type_ + ")" + "/kernel");
  kernel_.reset(K);
  this->parameters_[0] = kernel_; // add parameter
  if (bias_) {
    Tensor *b = stensor::zeros({out_channels_}, device, true);
    b_.reset(b);
    b_->set_name(name_ + "(" + type_ + ")" + "/bias");
    this->parameters_[1] = b_;// add parameter
  }
  InitCuDnn();

  handles_setup_ = true;
}

Conv2d::~Conv2d() {
  if (!handles_setup_) { return; }
  cudnnDestroyTensorDescriptor(input_desc_);
  cudnnDestroyTensorDescriptor(output_desc_);
  cudnnDestroyConvolutionDescriptor(conv_desc_);
  if (bias_) cudnnDestroyTensorDescriptor(bias_desc_);
  cudnnDestroyFilterDescriptor(filter_desc_);
  cudaStreamDestroy(stream_);
  cudnnDestroy(handle_);
  cudaFree(workspaceData);
}
}
}