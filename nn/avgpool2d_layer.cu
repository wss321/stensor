//
// Created by wss on 2022/4/5.
//
#include "avgpool2d_layer.hpp"
#include "math/math_base_cuda.hpp"

namespace stensor {
namespace nn {

template<typename Dtype>
__global__ void AvgPoolForward(const int nthreads,
                               const Dtype *const bottom_data, const int num, const int channels,
                               const int height, const int width, const int pooled_height,
                               const int pooled_width, const int kernel_h, const int kernel_w,
                               const int stride_h, const int stride_w, const int pad_h, const int pad_w,
                               Dtype *const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    const Dtype *const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    top_data[index] = aveval / pool_size;
  }
}

void AvgPool2d::forward_gpu() {
  const float *bottom_data = inputs_[0]->const_data();
  float *top_data = outputs_[0]->data();
  int count = outputs_[0]->size();
  // NOLINT_NEXT_LINE(whitespace/operators)
  AvgPoolForward<float><<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
      count, bottom_data, inputs_[0]->shape(0), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, padding_h_, padding_w_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template<typename Dtype>
__global__ void AvgPoolBackward(const int nthreads, const Dtype *const top_diff,
                                const int num, const int channels, const int height,
                                const int width, const int pooled_height, const int pooled_width,
                                const int kernel_h, const int kernel_w, const int stride_h,
                                const int stride_w, const int pad_h, const int pad_w,
                                Dtype *const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype *const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}

void AvgPool2d::backward_gpu() {
  if (!inputs_[0]->require_grad()) return;
  const float *top_diff = outputs_[0]->const_grad();
  float *bottom_diff = inputs_[0]->grad();
  const int count = inputs_[0]->size();
  gpu_set(count, float(0.), bottom_diff);

  AvgPoolBackward<float><<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
      count, top_diff, outputs_[0]->shape(0), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, padding_h_, padding_w_, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}
};//namespace nn
}// namespace stensor