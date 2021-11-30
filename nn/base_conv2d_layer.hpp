/**
* Copyright 2021 wss
* Created by wss on 11月,30, 2021
*/
#ifndef STENSOR_NN_BASE_CONV2D_LAYER_HPP_
#define STENSOR_NN_BASE_CONV2D_LAYER_HPP_
#include "module.hpp"
namespace stensor {

namespace nn {
class BaseConv2d : public Module {
 public:
  explicit BaseConv2d(const std::string &name,
                      int in_channels, int out_channels,
                      int kernel_h, int kernel_w,
                      int stride_h, int stride_w,
                      int padding_h, int padding_w,
                      int dilation_h, int dilation_w,
                      int group = 1,
                      int device = 0,
                      bool bias = true);
  ~BaseConv2d() override = default;;
  inline TensorVec forward(TensorVec &inputs) override {
    if (state_ == GPU)
      return forward_gpu(inputs);
    else
      return forward_cpu(inputs);
  };
  inline void backward() override {
    if (state_ == GPU)
      backward_gpu();
    else
      backward_cpu();
  }
 private:
  TensorVec forward_cpu(TensorVec &inputs);
  TensorVec forward_gpu(TensorVec &inputs);
  void backward_cpu();
  void backward_gpu();
  inline std::vector<int> calc_out_shape(const std::vector<int> &in_shape) {
    // B C H W
    CHECK_EQ(in_shape.size(), 4) << "Expected shape has 4 axes:[B, C, H, W]";
    int ow = (in_shape[3] - kernel_w_ + 2 * padding_w_) / stride_w_ + 1;
    int oh = (in_shape[2] - kernel_h_ + 2 * padding_h_) / stride_h_ + 1;
    return {in_shape[0], out_channels_, oh, ow};
  }

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
  SharedTensor col_buf_;
  SharedTensor bias_multiplier_;
  int conv_out_channels_;
  int conv_out_spatial_dim_{};
  int out_spatial_dim_{};
  int out_dim_{};
  int in_dim_{};
  int kernel_dim_;
  int weight_offset_;
  int col_offset_{};
  int output_offset_{};
 DISABLE_COPY_AND_ASSIGN(BaseConv2d);
};

}//namespace nn

}//namespace stensor
#endif //STENSOR_NN_BASE_CONV2D_LAYER_HPP_
