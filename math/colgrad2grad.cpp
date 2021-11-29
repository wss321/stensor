/**
* Copyright 2021 wss
* Created by wss on 11月,28, 2021
*/
#include "public/common.hpp"
#include "math_base_cpu.hpp"
namespace stensor {

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template<typename Dtype>
void cpu_colgrad2grad(const Dtype *data_col, const int channels,
                      const int height, const int width, const int kernel_h, const int kernel_w,
                      const int pad_h, const int pad_w,
                      const int stride_h, const int stride_w,
                      const int dilation_h, const int dilation_w,
                      Dtype *data_im) {
//  cpu_set(height * width * channels, Dtype(0), data_im);
  const int output_h = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w;
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

// Explicit instantiation
template void cpu_colgrad2grad<float>(const float *data_col, const int channels,
                                      const int height, const int width, const int kernel_h, const int kernel_w,
                                      const int pad_h, const int pad_w, const int stride_h,
                                      const int stride_w, const int dilation_h, const int dilation_w,
                                      float *data_im);
template void cpu_colgrad2grad<double>(const double *data_col, const int channels,
                                  const int height, const int width, const int kernel_h, const int kernel_w,
                                  const int pad_h, const int pad_w, const int stride_h,
                                  const int stride_w, const int dilation_h, const int dilation_w,
                                  double *data_im);

template<typename Dtype>
void cpu_colgrad2grad(const Dtype *data_col, const int num_spatial_axes,
                      const int *im_shape, const int *col_shape,
                      const int *kernel_shape, const int *pad, const int *stride,
                      const int *dilation, Dtype *data_im, const bool forced_3d) {
  const bool kIm2Col = false;
  cpu_ndimg_col_core(data_col, kIm2Col, num_spatial_axes, im_shape, col_shape,
                     kernel_shape, pad, stride, dilation, data_im, forced_3d);
}

// Explicit instantiation
template void cpu_colgrad2grad<float>(const float *data_col,
                                 const int num_spatial_axes,
                                 const int *im_shape, const int *col_shape,
                                 const int *kernel_shape, const int *pad, const int *stride,
                                 const int *dilation, float *data_im, const bool forced_3d);
template void cpu_colgrad2grad<double>(const double *data_col,
                                  const int num_spatial_axes,
                                  const int *im_shape, const int *col_shape,
                                  const int *kernel_shape, const int *pad, const int *stride,
                                  const int *dilation, double *data_im, const bool forced_3d);

}
