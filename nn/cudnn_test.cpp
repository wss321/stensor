/**
* Copyright 2021 wss
* Created by wss on 11月,29, 2021
*/
#include "public/common.hpp"
#include "core/math_tesnsor.hpp"
#include <gtest/gtest.h>
#include <cudnn.h>
#include <opencv2/opencv.hpp>
#include "math/math_base_cpu.hpp"
#include "core/tensor.hpp"
#include "io/image.hpp"

namespace stensor {
#define CUDNN_CHECK(condition) \
  do { \
    cudnnStatus_t status = condition; \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << " "\
      << cudnnGetErrorString(status); \
  } while (0)

using namespace cv;
class CudnnTest : public ::testing::Test {};
TEST_F(CudnnTest, conv2d) {
  std::string path = "/home/wss/CLionProjects/stensor/data/images/lena.png";
  int w = 64, h = 64;
  bool channel_first = true;
  stensor::Tensor *t_img = stensor::io::read_image2tensor(path, h, w, nullptr, channel_first);

//handle
  cudnnHandle_t handle;
  cudnnCreate(&handle);

// input
  t_img->to_gpu();
  t_img->reshape({1, t_img->shape(0),
                  t_img->shape(1),
                  t_img->shape(2)});
  cudnnTensorDescriptor_t input_descriptor;
  cudnnCreateTensorDescriptor(&input_descriptor);
  cudnnSetTensor4dDescriptor(input_descriptor,
                             CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT,
                             t_img->shape(0),
                             t_img->shape(1),
                             t_img->shape(2),
                             t_img->shape(3));

// output
  Tensor output(t_img->shape(), 0);

  cudnnTensorDescriptor_t output_descriptor;
  cudnnCreateTensorDescriptor(&output_descriptor);
  cudnnSetTensor4dDescriptor(output_descriptor,
                             CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT,
                             output.shape(0),
                             output.shape(1),
                             output.shape(2),
                             output.shape(3));

// kernel
  Tensor kernel({3, 3, 3, 3}, 0);
  int kernel_size = 9;//kernel.count(2, 4);
  int kernel_count = 9;
  float kernel_[9] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
  for (auto i = 0; i < kernel_count; ++i) {
    cpu_copy(9, kernel_, kernel.data() + i * kernel_size);
  }
  std::cout << kernel;
  cudnnFilterDescriptor_t kernel_descriptor;
  cudnnCreateFilterDescriptor(&kernel_descriptor);

  cudnnSetFilter4dDescriptor(kernel_descriptor,
                             CUDNN_DATA_FLOAT,
                             CUDNN_TENSOR_NCHW,
                             kernel.shape(0),
                             kernel.shape(1),
                             kernel.shape(2),
                             kernel.shape(3));
// convolution descriptor
  cudnnConvolutionDescriptor_t conv_descriptor;
  cudnnCreateConvolutionDescriptor(&conv_descriptor);
  cudnnSetConvolution2dDescriptor(conv_descriptor,
                                  1, 1, // zero-padding
                                  1, 1, // stride
                                  1, 1,
                                  CUDNN_CROSS_CORRELATION,
                                  CUDNN_DATA_FLOAT);
//  cudnnSetConvolutionGroupCount(conv_descriptor, 3);

// algorithm
  cudnnConvolutionFwdAlgo_t algo;
  cudnnGetConvolutionForwardAlgorithm(handle,
                                      input_descriptor,
                                      kernel_descriptor,
                                      conv_descriptor,
                                      output_descriptor,
                                      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                      0,
                                      &algo);

// workspace size && allocate memory
  size_t workspace_size = 0;
  cudnnGetConvolutionForwardWorkspaceSize(handle,
                                          input_descriptor,
                                          kernel_descriptor,
                                          conv_descriptor,
                                          output_descriptor,
                                          algo,
                                          &workspace_size);

  void *workspace = nullptr;
  cudaMalloc(&workspace, workspace_size);

// convolution
  auto alpha = 1.0f, beta = 0.0f;
  cudnnConvolutionForward(handle,
                          &alpha, input_descriptor, t_img->data(),
                          kernel_descriptor, kernel.data(),
                          conv_descriptor, algo,
                          workspace, workspace_size,
                          &beta, output_descriptor, output.data());

  cudnnGetConvolutionForwardWorkspaceSize(handle,
                                          input_descriptor,
                                          kernel_descriptor,
                                          conv_descriptor,
                                          output_descriptor,
                                          algo,
                                          &workspace_size);
  // choose backward algorithm for filter
  size_t workspace_limit_bytes = 8 * 1024 * 1024;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
  CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(handle,
                                                         input_descriptor,
                                                         output_descriptor,
                                                         conv_descriptor,
                                                         kernel_descriptor,
                                                         CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                         workspace_limit_bytes,
                                                         &bwd_filter_algo_));

  output.to_cpu();

// destroy
  cudaFree(workspace);

  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyConvolutionDescriptor(conv_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);

  cudnnDestroy(handle);


// show
  output.reshape({output.shape(1), output.shape(2), output.shape(3)});
  cv::Mat output_image = stensor::io::convertTensor2Mat(&output, channel_first);
  LOG(INFO) << output_image.rows << " " << output_image.cols << " " << output_image.channels();
  imshow("output", output_image);
  std::cout << output;

  waitKey();
  delete t_img;
}
}