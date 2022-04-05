/**
* Copyright 2021 wss
* Created by wss on 11月,15, 2021
*/
#ifndef STENSOR_CORE_MATH_BASE_CPU_HPP_
#define STENSOR_CORE_MATH_BASE_CPU_HPP_
#include <cblas.h>
#include <cmath>
#include <random>
#include <cstdint>
#include "public/common.hpp"
#include "omp.h"

namespace stensor {

/* self op start*/
template<typename Dtype>
void cpu_exp(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void cpu_log(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void cpu_abs(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void cpu_sqrt(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
Dtype cpu_asum(int n, const Dtype *x);
template<typename Dtype>
Dtype cpu_sum(int n, const Dtype *x);

template<typename Dtype>
void cpu_square(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void cpu_sign(const int n, const Dtype *x, Dtype *y);
template<typename Dtype>
void cpu_clamp(const int n,
               const Dtype min, const Dtype max,
               const Dtype *x,
               Dtype *y);
template<typename Dtype>
void cpu_sigmoid(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void cpu_tanh(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void cpu_relu(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void cpu_elu(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void cpu_gelu(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void cpu_leakyrelu(const int n, const Dtype *x, Dtype *y);

// x:MxDxN -> y:Mx1xN (sum at the second axis)
template<typename Dtype>
void cpu_reduce_sum(const int M, const int D, const int N, const Dtype *x, Dtype beta, Dtype *y);

// x:MxDxN -> y:Mx1xN (mean at the second axis)
template<typename Dtype>
void cpu_reduce_mean(const int M, const int D, const int N, const Dtype *x, Dtype beta, Dtype *y);

// x:MxDxN -> y:Mx1xN (variance at the second axis)
template<typename Dtype>
void cpu_reduce_var(const int M, const int D, const int N, const Dtype *x, Dtype beta, Dtype *y);

// x:MxDxN -> y:Mx1xN (Standard Deviation at the second axis)
template<typename Dtype>
void cpu_reduce_std(const int M, const int D, const int N, const Dtype *x, Dtype beta, Dtype *y);

// x:MxDxN -> y:Mx1xN (asum at the second axis)
template<typename Dtype>
void cpu_reduce_asum(const int M, const int D, const int N, const Dtype *x, Dtype beta, Dtype *y);

// x:MxDxN -> y:MxDxN (softmax at the second axis)
template<typename Dtype>
void cpu_softmax(const int M, const int D, const int N, const Dtype *x, Dtype beta, Dtype *y);

// x:M -> y:MxC (one_hot)
template<typename Dtype>
void cpu_one_hot(const int M, const int C, const Dtype *x, Dtype *y);

template<typename Dtype>
void cpu_one_hot(const int M, const int C, const int *x, Dtype *y);
template<typename Dtype>
void cpu_argmax(const int M, const int D, const int N, const Dtype *x, Dtype *y);

/* self op end*/

/* vector scalar start*/
template<typename Dtype>
void cpu_set(const int n,
             const Dtype val,
             Dtype *y);

template<typename Dtype>
void cpu_add_scalar(const int n,
                    const Dtype *x, const Dtype val,
                    Dtype *y);

template<typename Dtype>
void cpu_sub_scalar(const int n,
                    const Dtype *x, const Dtype val,
                    Dtype *y);

template<typename Dtype>
void cpu_scale(const int n,
               const Dtype *x, const Dtype val,
               Dtype *y);

template<typename Dtype>
void cpu_pow_scalar(const int n,
                    const Dtype *a, const Dtype val,
                    Dtype *y);

/* vector scalar end*/

/* vector vector start*/
template<typename Dtype>
bool cpu_equal(const int n,
               const Dtype *a,
               const Dtype *b);

template<typename Dtype>
void cpu_copy(const int n,
              const Dtype *a,
              Dtype *y);

template<typename Dtype>
void cpu_add(const int n,
             const Dtype *a, const Dtype *b,
             Dtype *y);

template<typename Dtype>
void cpu_sub(const int n,
             const Dtype *a, const Dtype *b,
             Dtype *y);

template<typename Dtype>
void cpu_mul(const int n,
             const Dtype *x, const Dtype *b,
             Dtype *y);

template<typename Dtype>
void cpu_div(const int n,
             const Dtype *a, const Dtype *b,
             Dtype *y);
template<typename Dtype>
void cpu_pow(const int n,
             const Dtype *a, const Dtype *b,
             Dtype *y);
template<typename Dtype>
void cpu_add_broadcast(const Dtype *a, const Dtype *b,
                       std::vector<int> &shape_a,
                       std::vector<int> &shape_b,
                       Dtype *y);

template<typename Dtype>
void cpu_sub_broadcast(const Dtype *a, const Dtype *b,
                       std::vector<int> &shape_a,
                       std::vector<int> &shape_b,
                       Dtype *y);

template<typename Dtype>
void cpu_mul_broadcast(const Dtype *a, const Dtype *b,
                       std::vector<int> &shape_a,
                       std::vector<int> &shape_b,
                       Dtype *y);

template<typename Dtype>
void cpu_div_broadcast(const Dtype *a, const Dtype *b,
                       std::vector<int> &shape_a,
                       std::vector<int> &shape_b,
                       Dtype *y);

template<typename Dtype>
void cpu_pow_broadcast(const Dtype *a, const Dtype *b,
                       std::vector<int> &shape_a,
                       std::vector<int> &shape_b,
                       Dtype *y);

template<typename Dtype>
void cpu_maximum(const int n,
                 const Dtype *a, const Dtype *b,
                 Dtype *y);

template<typename Dtype>
void cpu_minimum(const int n,
                 const Dtype *a, const Dtype *b,
                 Dtype *y);

// Returns the sum of the absolute values of the elements of vector a

template<typename Dtype>
Dtype cpu_dot(int n, const Dtype *a, const Dtype *b);

template<typename Dtype>
Dtype cpu_stride_dot(int n,
                     const Dtype *a, int incx,
                     const Dtype *y, int incy);

/* vector-vector end*/

/* matrix vector start*/
template<typename Dtype>
void cpu_gemv(const bool TransA,
              const int M, const int N,
              const Dtype alpha, const Dtype *A, const Dtype *a,
              const Dtype beta, Dtype *y);

/* matrix-vector end*/

/* matrix-matrix start*/
template<typename Dtype>
void cpu_gemm(const bool TransA,
              const bool TransB,
              const int M, const int N, const int K,
              const Dtype alpha, const Dtype *A, const Dtype *B,
              const Dtype beta, Dtype *C);

template<typename Dtype>
void cpu_axpy(const int n,
              const Dtype alpha, const Dtype *a,
              Dtype *y);

template<typename Dtype>
void cpu_axpby(const int n,
               const Dtype alpha, const Dtype *a,
               const Dtype beta, Dtype *y);

/* matrix-matrix end*/

/* random generator start*/
template<typename Dtype>
void cpu_rng_uniform(int n,
                     Dtype a, Dtype b,
                     Dtype *r);

template<typename Dtype>
void cpu_rng_gaussian(int n,
                      Dtype mu, Dtype sigma,
                      Dtype *r);

template<typename Dtype>
void cpu_rng_bernoulli(int n,
                       Dtype p,
                       int *r);
/* random generator end*/

// loss
// x:n -> y:n
template<typename Dtype>
void cpu_nll(const int n, const Dtype *pred, const Dtype *ground_truth, Dtype *y);

//conv
template<typename Dtype>
void cpu_img2col(const int n, int c, int h, int w, int stride_w, int stride_h,
                 const Dtype *img, Dtype *col);

template<typename Dtype>
void cpu_ndimg2col(const Dtype *data_im, const int num_spatial_axes,
                   const int *im_shape, const int *col_shape,
                   const int *kernel_shape, const int *pad, const int *stride,
                   const int *dilation, Dtype *data_col, const bool forced_3d = false);

template<typename Dtype>
void cpu_img2col(const Dtype *data_im, const int channels,
                 const int height, const int width, const int kernel_h, const int kernel_w,
                 const int pad_h, const int pad_w, const int stride_h,
                 const int stride_w, const int dilation_h, const int dilation_w,
                 Dtype *data_col);

template<typename Dtype>
void cpu_colgrad2grad(const Dtype *data_col, const int num_spatial_axes,
                      const int *im_shape, const int *col_shape,
                      const int *kernel_shape, const int *pad, const int *stride,
                      const int *dilation, Dtype *data_im, const bool forced_3d = false);

template<typename Dtype>
void cpu_colgrad2grad(const Dtype *data_col, const int channels,
                      const int height, const int width, const int kernel_h, const int kernel_w,
                      const int pad_h, const int pad_w, const int stride_h,
                      const int stride_w, const int dilation_h, const int dilation_w,
                      Dtype *data_im);
template<typename Dtype>
inline void cpu_ndimg_col_core(const Dtype *data_input, const bool im2col,
                               const int num_spatial_axes, const int *im_shape, const int *col_shape,
                               const int *kernel_shape, const int *pad, const int *stride,
                               const int *dilation, Dtype *data_output, const bool forced_3d);

}
#endif //STENSOR_CORE_MATH_BASE_CPU_HPP_
