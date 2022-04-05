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
void cpu_img2col(const Dtype *data_im, const int channels,
                 const int height, const int width, const int kernel_h, const int kernel_w,
                 const int pad_h, const int pad_w,
                 const int stride_h, const int stride_w,
                 const int dilation_h, const int dilation_w,
                 Dtype *data_col) {
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
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
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
template void cpu_img2col<float>(const float *data_im, const int channels,
                                const int height, const int width, const int kernel_h, const int kernel_w,
                                const int pad_h, const int pad_w, const int stride_h,
                                const int stride_w, const int dilation_h, const int dilation_w,
                                float *data_col);
template void cpu_img2col<double>(const double *data_im, const int channels,
                                 const int height, const int width, const int kernel_h, const int kernel_w,
                                 const int pad_h, const int pad_w, const int stride_h,
                                 const int stride_w, const int dilation_h, const int dilation_w,
                                 double *data_col);

template<typename Dtype>
inline void cpu_ndimg_col_core(const Dtype *data_input, const bool im2col,
                               const int num_spatial_axes, const int *im_shape, const int *col_shape,
                               const int *kernel_shape, const int *pad, const int *stride,
                               const int *dilation, Dtype *data_output, const bool forced_3d) {
  for (int i = 0; i < num_spatial_axes; ++i) {
  }
  if (!im2col) {
    int im_size = im_shape[0];
    for (int i = 0; i < num_spatial_axes; ++i) {
      im_size *= im_shape[1 + i + forced_3d];
    }
//    cpu_set(im_size, Dtype(0), data_output);
  }
  int kernel_size = 1;
  for (int i = 0; i < num_spatial_axes; ++i) {
    kernel_size *= kernel_shape[i];
  }
  const int channels_col = col_shape[0];
  std::vector<int> d_offset(num_spatial_axes, 0);
  std::vector<int> d_iter(num_spatial_axes, 0);
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    // Loop over spatial axes in reverse order to compute a per-axis offset.
    int offset = c_col;
    for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
      if (d_i < num_spatial_axes - 1) {
        offset /= kernel_shape[d_i + 1];
      }
      d_offset[d_i] = offset % kernel_shape[d_i];
    }
    for (bool incremented = true; incremented;) {
      // Loop over spatial axes in forward order to compute the indices in the
      // image and column, and whether the index lies in the padding.
      int index_col = c_col;
      int index_im = c_col / kernel_size;
      bool is_padding = false;
      for (int d_i = 0; d_i < num_spatial_axes; ++d_i) {
        const int d = d_iter[d_i];
        const int d_im = d * stride[d_i] - pad[d_i] +
            d_offset[d_i] * dilation[d_i];
        is_padding |= d_im < 0 || d_im >= im_shape[d_i + 1 + forced_3d];
        index_col *= col_shape[d_i + 1 + forced_3d];
        index_col += d;
        index_im *= im_shape[d_i + 1 + forced_3d];
        index_im += d_im;
      }
      if (im2col) {
        if (is_padding) {
          data_output[index_col] = 0;
        } else {
          data_output[index_col] = data_input[index_im];
        }
      } else if (!is_padding) {  // col2im
        data_output[index_im] += data_input[index_col];
      }
      // Loop over spatial axes in reverse order to choose an index,
      // like counting.
      incremented = false;
      for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
        const int d_max = col_shape[d_i + 1 + forced_3d];
        DCHECK_LT(d_iter[d_i], d_max);
        if (d_iter[d_i] == d_max - 1) {
          d_iter[d_i] = 0;
        } else {  // d_iter[d_i] < d_max - 1
          ++d_iter[d_i];
          incremented = true;
          break;
        }
      }
    }  // while(incremented) {
  }  // for (int c = 0; c < channels_col; ++c) {
}

template<typename Dtype>
void cpu_ndimg2col(const Dtype *data_im, const int num_spatial_axes,
                   const int *im_shape, const int *col_shape,
                   const int *kernel_shape, const int *pad, const int *stride,
                   const int *dilation, Dtype *data_col, const bool forced_3d) {
  const bool kIm2Col = true;
  cpu_ndimg_col_core(data_im, kIm2Col, num_spatial_axes, im_shape, col_shape,
                     kernel_shape, pad, stride, dilation, data_col, forced_3d);
}

// Explicit instantiation
template void cpu_ndimg2col<float>(const float *data_im,
                                   const int num_spatial_axes,
                                   const int *im_shape, const int *col_shape,
                                   const int *kernel_shape, const int *pad, const int *stride,
                                   const int *dilation, float *data_col, const bool forced_3d);
template void cpu_ndimg2col<double>(const double *data_im,
                                    const int num_spatial_axes,
                                    const int *im_shape, const int *col_shape,
                                    const int *kernel_shape, const int *pad, const int *stride,
                                    const int *dilation, double *data_col, const bool forced_3d);

}