/**
* Copyright 2021 wss
* Created by wss on 11月,29, 2021
*/
#include "conv2d_layer.hpp"

namespace stensor {
namespace nn {

float dataType<float>::oneval = 1.0;
float dataType<float>::zeroval = 0.0;
const void* dataType<float>::one =
    static_cast<void *>(&dataType<float>::oneval);
const void* dataType<float>::zero =
    static_cast<void *>(&dataType<float>::zeroval);

double dataType<double>::oneval = 1.0;
double dataType<double>::zeroval = 0.0;
const void* dataType<double>::one =
    static_cast<void *>(&dataType<double>::oneval);
const void* dataType<double>::zero =
    static_cast<void *>(&dataType<double>::zeroval);


void Conv2d::setUpConvAlg(const std::vector<int> &in_shape) {
  std::vector<int> out_shape = calc_out_shape(in_shape);
  cudnnCreateTensorDescriptor(&input_desc_);
  cudnnSetTensor4dDescriptor(input_desc_,
                             CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT,
                             in_shape[0],
                             in_shape[1],
                             in_shape[2],
                             in_shape[3]);
  cudnnCreateTensorDescriptor(&output_desc_);
  cudnnSetTensor4dDescriptor(output_desc_,
                             CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT,
                             out_shape[0],
                             out_shape[1],
                             out_shape[2],
                             out_shape[3]);
  size_t workspace_limit_bytes = 64 * 1024 * 1024;
//   choose forward and backward algorithms + workspace(s)
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_,
                                                  input_desc_,
                                                  filter_desc_,
                                                  conv_desc_,
                                                  output_desc_,
                                                  CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                  workspace_limit_bytes,
                                                  &fwd_algo_));
//  fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_,
                                                      input_desc_,
                                                      filter_desc_,
                                                      conv_desc_,
                                                      output_desc_,
                                                      fwd_algo_,
                                                      &workspace_fwd_sizes_));
//  fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
//  bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
//  bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

  // choose backward algorithm for filter
  CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(handle_,
                                                         input_desc_, output_desc_, conv_desc_, filter_desc_,
                                                         CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                         workspace_limit_bytes, &bwd_filter_algo_));

  // get workspace for backwards filter algorithm
  CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_,
                                                             input_desc_, output_desc_, conv_desc_, filter_desc_,
                                                             bwd_filter_algo_, &workspace_bwd_filter_sizes_));

  // choose backward algo for data
  CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(handle_,
                                                       filter_desc_, output_desc_, conv_desc_, input_desc_,
                                                       CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                       workspace_limit_bytes, &bwd_data_algo_));

  // get workspace size
  CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle_,
                                                           filter_desc_, output_desc_, conv_desc_, input_desc_,
                                                           bwd_data_algo_, &workspace_bwd_data_sizes_));
  size_t max_workspace = std::max(workspace_fwd_sizes_,
                                  workspace_bwd_data_sizes_);
  max_workspace = std::max(max_workspace, workspace_bwd_filter_sizes_);
  if (max_workspace > workspaceSizeInBytes) {
//    DLOG(INFO) << "Reallocating workspace storage: " << max_workspace;
    workspaceSizeInBytes = max_workspace;
    cudaFree(workspaceData);
    cudaError_t err = cudaMalloc(&(this->workspaceData), workspaceSizeInBytes);
    if (err != cudaSuccess) {
      // force zero memory path
      workspace_fwd_sizes_ = 0;
      workspace_bwd_filter_sizes_ = 0;
      workspace_bwd_data_sizes_ = 0;
      fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
      bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
      bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
      workspace = nullptr;
      workspaceSizeInBytes = 0;
    }
    workspace = reinterpret_cast<char *>(workspaceData);
  }
}

TensorVec Conv2d::forward(TensorVec &inputs) {
  CHECK_EQ(inputs.size(), 1) << "Only support one input tensor now";
  this->zero_output_grad();
  SharedTensor in = inputs[0];
  std::vector<int> out_shape(calc_out_shape(in->shape()));
  const float *weight = kernel_->data();
  const float *in_data = in->const_data();
  if (result_.get() == nullptr || out_shape != result_->shape()) {
    outputs_.resize(1);
    result_.reset(new Tensor(out_shape, in->device(), kernel_->require_grad() || in->require_grad()));
    result_->set_name(name() + "/output");
    setUpConvAlg(in->shape());
    outputs_[0] = result_;
  }
  inputs_ = inputs;
  CUDNN_CHECK(cudnnConvolutionForward(handle_,
                                      dataType<float>::one,
                                      input_desc_, in_data,
                                      filter_desc_, weight,
                                      conv_desc_,
                                      fwd_algo_, workspace, workspace_fwd_sizes_,
                                      dataType<float>::zero,
                                      output_desc_, result_->data()));
  if (bias_) {
    const float *bias_data = b_->data();
    CUDNN_CHECK(cudnnAddTensor(handle_,
                               dataType<float>::one,
                               bias_desc_, bias_data,
                               dataType<float>::one,
                               output_desc_, result_->data()));
  }
  return outputs_;
}

void nn::Conv2d::backward() {
  const float *weight = kernel_->data();
  float *weight_grad = kernel_->grad();
  const float *top_diff = result_->grad();
  if (bias_ && b_->require_grad()) {
    CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_,
                                             dataType<float>::one,
                                             output_desc_, top_diff,
                                             dataType<float>::one,
                                             bias_desc_, b_->grad()));

  }
  if (kernel_->require_grad()) {
    const float *bottom_data = inputs_[0]->data();
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(
        handle_,
        dataType<float>::one,
        input_desc_, bottom_data,
        output_desc_, top_diff,
        conv_desc_,
        bwd_filter_algo_, workspace,
        workspace_bwd_filter_sizes_,
        dataType<float>::one,
        filter_desc_, weight_grad));
  }
  if (inputs_[0]->require_grad()) {
    float *bottom_diff = inputs_[0]->grad();
    CUDNN_CHECK(cudnnConvolutionBackwardData(
        handle_,
        dataType<float>::one,
        filter_desc_, weight,
        output_desc_, top_diff,
        conv_desc_,
        bwd_data_algo_, workspace,
        workspace_bwd_data_sizes_,
        dataType<float>::one,
        input_desc_, bottom_diff));
  }
  inputs_.clear();
}

}
}