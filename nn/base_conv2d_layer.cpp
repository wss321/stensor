/**
* Copyright 2021 wss
* Created by wss on 11月,30, 2021
*/
#include "base_conv2d_layer.hpp"
#include "math/math_base_cpu.hpp"
#include "core/math_tesnsor.hpp"
namespace stensor {

namespace nn {

BaseConv2d::BaseConv2d(const std::string &name,
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
  conv_out_channels_ = out_channels_;
  kernel_dim_ = kernel_w * kernel_h * in_channels;
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  if (bias_) {
    Tensor *b = stensor::zeros({out_channels_}, device, true);
    b_.reset(b);
    b_->set_name(name_ + "(" + type_ + ")" + "/bias");
    this->parameters_[1] = b_;// add parameter
  }
}

TensorVec BaseConv2d::forward_cpu(TensorVec &inputs) {
  inputs_.clear();
  CHECK_EQ(inputs.size(), 1) << "Only support one input tensor now";
  this->zero_output_grad();
  SharedTensor in = inputs[0];
  std::vector<int> out_shape(calc_out_shape(in->shape()));
  if (result_.get() == nullptr || out_shape != result_->shape()) {
    outputs_.resize(1);
    result_.reset(new Tensor(out_shape, in->device(), kernel_->require_grad() || in->require_grad()));
    result_->set_name(name() + "/output");
    outputs_[0] = result_;
    std::vector<int> col_shape(2);
    col_shape[0] = group_ * kernel_->count(1, kernel_->num_axes());
    col_shape[1] = out_shape[2] * out_shape[3];
    col_buf_.reset(new Tensor(col_shape, in->device(), false));
    conv_out_spatial_dim_ = outputs_[0]->count(2, outputs_[0]->num_axes());

    col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
    output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
    out_spatial_dim_ = result_->count(2, result_->num_axes());
    bias_multiplier_.reset(stensor::ones({1, out_spatial_dim_}, in->device(), false));
    in_dim_ = in->count(1, in->num_axes());
    out_dim_ = result_->count(1, result_->num_axes());
  }
  inputs_ = inputs;
  const float *bottom_data = in->const_data();
  const float *weights = kernel_->const_data();
  float *output = outputs_[0]->data();
  int h = in->shape(2);
  int w = in->shape(3);

  for (int n = 0; n < in->shape(0); ++n) {//batch
    float *col_buff = col_buf_->data();
    cpu_img2col(bottom_data,
                in_channels_,
                h, w,
                kernel_h_, kernel_w_,
                padding_h_, padding_w_,
                stride_h_, stride_w_,
                dilation_h_, dilation_w_,
                col_buff);
    for (int g = 0; g < group_; ++g) {
      cpu_gemm(false, false, conv_out_channels_ / group_, conv_out_spatial_dim_, kernel_dim_,
               (float) 1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
               (float) 0., output + output_offset_ * g);
    }
    if (bias_) {
      const float *bias = b_->const_data();
      cpu_gemm(false, false, out_channels_,
               out_spatial_dim_, 1, (float) 1., bias, bias_multiplier_->const_data(),
               (float) 1., output);
    }
    bottom_data += in_dim_;
    output += out_dim_;
  }
  return outputs_;
}

void BaseConv2d::backward_cpu() {
  const float *weights = kernel_->const_data();
  float *weight_diff = kernel_->grad();
  SharedTensor in = inputs_[0];
  int h = in->shape(2);
  int w = in->shape(3);
  if (bias_ && b_->require_grad()) {
    const float *top_diff = result_->const_grad();
    float *bias_diff = b_->grad();
    for (int n = 0; n < result_->shape(0); ++n) {
      cpu_gemv(false, conv_out_channels_, out_spatial_dim_, (float) 1.,
               top_diff, bias_multiplier_->const_data(), (float) 1., bias_diff);
      top_diff += out_dim_;
    }
  }

  if (kernel_->require_grad() || inputs_[0]->require_grad()) {
    const float *top_diff = result_->const_grad();
    const float *bottom_data = inputs_[0]->const_data();
    for (int n = 0; n < result_->shape(0); ++n) {
      float *col_buff = col_buf_->data();
      cpu_img2col(bottom_data,
                  in_channels_,
                  h, w,
                  kernel_h_, kernel_w_,
                  padding_h_, padding_w_,
                  stride_h_, stride_w_,
                  dilation_h_, dilation_w_,
                  col_buff);
      if (kernel_->require_grad())
        for (int g = 0; g < group_; ++g) {
          cpu_gemm(false, false, conv_out_channels_ / group_,
                   kernel_dim_, conv_out_spatial_dim_,
                   (float) 1., top_diff + output_offset_ * g, col_buff + col_offset_ * g,
                   (float) 1., weight_diff + weight_offset_ * g);
        }
      if (inputs_[0]->require_grad()) {
        float *bottom_diff = inputs_[0]->grad();
        for (int g = 0; g < group_; ++g) {
          cpu_gemm(false, false, kernel_dim_,
                   conv_out_spatial_dim_, conv_out_channels_ / group_,
                   (float) 1., weights + weight_offset_ * g, top_diff + output_offset_ * g,
                   (float) 0., col_buff + col_offset_ * g);
        }
        cpu_colgrad2grad(col_buff,
                         in_channels_,
                         h, w,
                         kernel_h_, kernel_w_,
                         padding_h_, padding_w_,
                         stride_h_, stride_w_,
                         dilation_h_, dilation_w_,
                         bottom_diff);
      }
      top_diff += out_dim_;
      bottom_data += in_dim_;
    }//n
  }

}

TensorVec BaseConv2d::forward_gpu(TensorVec &inputs) {
  inputs_.clear();
  CHECK_EQ(inputs.size(), 1) << "Only support one input tensor now";
  this->zero_output_grad();
  SharedTensor in = inputs[0];
  std::vector<int> out_shape(calc_out_shape(in->shape()));
  if (result_.get() == nullptr || out_shape != result_->shape()) {
    outputs_.resize(1);
    result_.reset(new Tensor(out_shape, in->device(), kernel_->require_grad() || in->require_grad()));
    result_->set_name(name() + "/output");
    outputs_[0] = result_;
    std::vector<int> col_shape(2);
    col_shape[0] = group_ * kernel_->count(1, kernel_->num_axes());
    col_shape[1] = out_shape[2] * out_shape[3];
    col_buf_.reset(new Tensor(col_shape, in->device(), false));
    conv_out_spatial_dim_ = outputs_[0]->count(2, outputs_[0]->num_axes());

    col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
    output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
    out_spatial_dim_ = result_->count(2, result_->num_axes());
    bias_multiplier_.reset(stensor::ones({1, out_spatial_dim_}, in->device(), false));
    in_dim_ = in->count(1, in->num_axes());
    out_dim_ = result_->count(1, result_->num_axes());
  }
  inputs_ = inputs;
  const float *bottom_data = in->const_data();
  const float *weights = kernel_->const_data();
  float *output = outputs_[0]->data();
  int h = in->shape(2);
  int w = in->shape(3);

  for (int n = 0; n < in->shape(0); ++n) {//batch
    float *col_buff = col_buf_->data();
    gpu_img2col(bottom_data,
                in_channels_,
                h, w,
                kernel_h_, kernel_w_,
                padding_h_, padding_w_,
                stride_h_, stride_w_,
                dilation_h_, dilation_w_,
                col_buff);
    for (int g = 0; g < group_; ++g) {
      gpu_gemm(false, false, conv_out_channels_ / group_, conv_out_spatial_dim_, kernel_dim_,
               (float) 1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
               (float) 0., output + output_offset_ * g);
    }
    if (bias_) {
      const float *bias = b_->const_data();
      gpu_gemm(false, false, out_channels_,
               out_spatial_dim_, 1, (float) 1., bias, bias_multiplier_->const_data(),
               (float) 1., output);
    }
    bottom_data += in_dim_;
    output += out_dim_;
  }
  return outputs_;
}
void BaseConv2d::backward_gpu() {
  const float *weights = kernel_->const_data();
  float *weight_diff = kernel_->grad();
  SharedTensor in = inputs_[0];
  int h = in->shape(2);
  int w = in->shape(3);
  if (bias_ && b_->require_grad()) {
    const float *top_diff = result_->const_grad();
    float *bias_diff = b_->grad();
    for (int n = 0; n < result_->shape(0); ++n) {
      gpu_gemv(false, conv_out_channels_, out_spatial_dim_, (float) 1.,
               top_diff, bias_multiplier_->const_data(), (float) 1., bias_diff);
      top_diff += out_dim_;
    }
  }

  if (kernel_->require_grad() || inputs_[0]->require_grad()) {
    const float *top_diff = result_->const_grad();
    const float *bottom_data = inputs_[0]->const_data();
    for (int n = 0; n < result_->shape(0); ++n) {
      float *col_buff = col_buf_->data();
      gpu_img2col(bottom_data,
                  in_channels_,
                  h, w,
                  kernel_h_, kernel_w_,
                  padding_h_, padding_w_,
                  stride_h_, stride_w_,
                  dilation_h_, dilation_w_,
                  col_buff);
      if (kernel_->require_grad())
        for (int g = 0; g < group_; ++g) {
          gpu_gemm(false, false, conv_out_channels_ / group_,
                   kernel_dim_, conv_out_spatial_dim_,
                   (float) 1., top_diff + output_offset_ * g, col_buff + col_offset_ * g,
                   (float) 1., weight_diff + weight_offset_ * g);
        }
      if (inputs_[0]->require_grad()) {
        float *bottom_diff = inputs_[0]->grad();
        for (int g = 0; g < group_; ++g) {
          gpu_gemm(false, false, kernel_dim_,
                   conv_out_spatial_dim_, conv_out_channels_ / group_,
                   (float) 1., weights + weight_offset_ * g, top_diff + output_offset_ * g,
                   (float) 0., col_buff + col_offset_ * g);
        }
        gpu_colgrad2grad(col_buff,
                         in_channels_,
                         h, w,
                         kernel_h_, kernel_w_,
                         padding_h_, padding_w_,
                         stride_h_, stride_w_,
                         dilation_h_, dilation_w_,
                         bottom_diff);
      }
      top_diff += out_dim_;
      bottom_data += in_dim_;
    }//n
  }
}

}//namespace nn

}//namespace stensor