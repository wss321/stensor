//
// Created by wss on 2022/4/5.
//
#include "avgpool2d_layer.hpp"
#include "math/math_base_cpu.hpp"

namespace stensor {
namespace nn {

TensorVec AvgPool2d::forward(TensorVec &inputs) {
  // set up input and output shape, then call forward cpu/gpu
  CHECK_EQ(inputs.size(), 1) << "Only support one input tensor now";
  this->zero_output_grad();
  SharedTensor in = inputs[0];
  height_ = in->shape(2);
  width_ = in->shape(3);
  channels_ = in->shape(1);
  std::vector<int> out_shape(calc_out_shape(in->shape()));
  if (result_.get() == nullptr || out_shape != result_->shape()) {
    outputs_.resize(1);
    result_.reset(new Tensor(out_shape, in->device(), in->require_grad()));
    result_->set_name(name() + "/output");
    max_idx_.reset(new Tensor(out_shape, in->device(), in->require_grad()));
    max_idx_->set_name(name() + "/max_idx");
    outputs_[0] = result_;
    pooled_height_ = out_shape[2];
    pooled_width_ = out_shape[3];
  }
  inputs_.clear();
  inputs_ = inputs;
  switch (state()) {
    case CPU:forward_cpu();
      break;
    case GPU:forward_gpu();
      break;
  }
  return outputs_;
}
void AvgPool2d::forward_cpu() {
  SharedTensor in = inputs_[0];
  const float *bottom_data = inputs_[0]->const_data();
  float *top_data = outputs_[0]->data();
  const int top_count = outputs_[0]->size();
  // Initialize
  for (int i = 0; i < top_count; ++i) {
    top_data[i] = 0;
  }
  // The main loop
  for (int n = 0; n < inputs_[0]->shape(0); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          int hstart = ph * stride_h_ - padding_h_;
          int wstart = pw * stride_w_ - padding_w_;
          int hend = std::min(hstart + kernel_h_, height_ + padding_h_);
          int wend = std::min(wstart + kernel_w_, width_ + padding_w_);
          int pool_size = (hend - hstart) * (wend - wstart);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          hend = std::min(hend, height_);
          wend = std::min(wend, width_);
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              top_data[ph * pooled_width_ + pw] +=
                  bottom_data[h * width_ + w];
            }
          }
          top_data[ph * pooled_width_ + pw] /= pool_size;
        }
      }
      // compute offset
      bottom_data += inputs_[0]->offset({0, 1});
      top_data += outputs_[0]->offset({0, 1});
    }
  }
}

void AvgPool2d::backward() {
  if (state() == GPU)
    backward_gpu();
  else
    backward_cpu();
}

void AvgPool2d::backward_cpu() {
  if (!inputs_[0]->require_grad()) return;
  const float *top_diff = outputs_[0]->const_grad();
  float *bottom_diff = inputs_[0]->grad();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  cpu_set(inputs_[0]->size(), float(0), bottom_diff);

  // The main loop
  for (int n = 0; n < outputs_[0]->shape(0); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          int hstart = ph * stride_h_ - padding_h_;
          int wstart = pw * stride_w_ - padding_w_;
          int hend = std::min(hstart + kernel_h_, height_ + padding_h_);
          int wend = std::min(wstart + kernel_w_, width_ + padding_w_);
          int pool_size = (hend - hstart) * (wend - wstart);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          hend = std::min(hend, height_);
          wend = std::min(wend, width_);
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              bottom_diff[h * width_ + w] +=
                  top_diff[ph * pooled_width_ + pw] / pool_size;
            }
          }
        }
      }
      // offset
      bottom_diff += inputs_[0]->offset({0, 1});
      top_diff += outputs_[0]->offset({0, 1});
    }
  }
}

};//namespace nn
}// namespace stensor