//
// Created by wss on 2022/4/5.
//
#include "maxpool2d_layer.hpp"
#include "math/math_base_cuda.hpp"

namespace stensor {
namespace nn {

template<typename Dtype>
__global__ void MaxPoolForward(const int nthreads,
                               const Dtype *const bottom_data, const int num, const int channels,
                               const int height, const int width, const int pooled_height,
                               const int pooled_width, const int kernel_h, const int kernel_w,
                               const int stride_h, const int stride_w, const int pad_h, const int pad_w,
                               Dtype *const top_data, float *mask, Dtype *top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype *const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

void MaxPool2d::forward_gpu() {
  const float *bottom_data = inputs_[0]->const_data();
  float *top_data = outputs_[0]->data();
  int count = outputs_[0]->size();
  // We'll output the mask to outputs_[1] if it's of size >1.
  const bool use_top_mask = outputs_.size() > 1;
  float *mask = nullptr;
  float *top_mask = nullptr;
  if (use_top_mask) {
    top_mask = outputs_[1]->data();
  } else {
    mask = max_idx_->data();
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  MaxPoolForward<float><<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
      count, bottom_data, inputs_[0]->shape(0), channels_,
      height_, width_, pooled_height_, pooled_width_, kernel_h_,
      kernel_w_, stride_h_, stride_w_, padding_h_, padding_w_, top_data,
      mask, top_mask);
  CUDA_POST_KERNEL_CHECK;
}

template<typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype *const top_diff,
                                const float *const mask, const Dtype *const top_mask, const int num,
                                const int channels, const int height, const int width,
                                const int pooled_height, const int pooled_width, const int kernel_h,
                                const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
                                const int pad_w, Dtype *const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const Dtype *const top_diff_slice = top_diff + offset;
    if (mask) {
      const float *const mask_slice = mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    } else {
      const Dtype *const top_mask_slice = top_mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

void MaxPool2d::backward_gpu() {
  if (!inputs_[0]->require_grad()) return;
  const float *top_diff = outputs_[0]->const_grad();
  float *bottom_diff = inputs_[0]->grad();
  const int count = inputs_[0]->size();
  gpu_set(count, float(0.), bottom_diff);
  // We'll output the mask to outputs_[1] if it's of size >1.
  const bool use_top_mask = outputs_.size() > 1;
  const float *mask = nullptr;
  const float *top_mask = nullptr;
  if (use_top_mask) {
    top_mask = outputs_[1]->const_data();
  } else {
    mask = max_idx_->const_data();
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  MaxPoolBackward<float><<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
      count, top_diff, mask, top_mask, outputs_[0]->shape(0), channels_,
      height_, width_, pooled_height_, pooled_width_,
      kernel_h_, kernel_w_, stride_h_, stride_w_, padding_h_, padding_w_,
      bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}
};//namespace nn
}// namespace stensor