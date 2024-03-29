/**
* Copyright 2021 wss
* Created by wss on 11月,16, 2021
*/
#ifndef STENSOR_CORE_MATH_BASE_CUDA_HPP_
#define STENSOR_CORE_MATH_BASE_CUDA_HPP_

#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cudnn.h>
#include <cblas-atlas.h>

#include "public/common.hpp"
#include "public/memory_op.hpp"
#include "public/stensor_random.hpp"

namespace stensor {
/* self-op start*/
template<typename Dtype>
void gpu_exp(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void gpu_log(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void gpu_abs(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void gpu_sqrt(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void gpu_square(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void gpu_asum(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void gpu_sign(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void gpu_sgnbit(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void gpu_clamp(const int n,
               const Dtype min, const Dtype max,
               const Dtype *x,
               Dtype *y);
template<typename Dtype>
void gpu_sigmoid(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void gpu_tanh(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void gpu_relu(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void gpu_elu(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void gpu_gelu(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void gpu_leakyrelu(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void gpu_reduce_sum(const int M, const int D, const int N, const Dtype *x, Dtype beta, Dtype *y);

template<typename Dtype>
void gpu_reduce_mean(const int M, const int D, const int N, const Dtype *x, Dtype beta, Dtype *y);

template<typename Dtype>
void gpu_reduce_asum(const int M, const int D, const int N, const Dtype *x, Dtype beta, Dtype *y);

template<typename Dtype>
void gpu_softmax(const int M, const int D, const int N, const Dtype *x, Dtype beta, Dtype *y);

// x:M -> y:MxC (one_hot)
template<typename Dtype>
void gpu_one_hot(const int M, const int C, const Dtype *x, Dtype *y);

template<typename Dtype>
void gpu_one_hot(const int M, const int C, const int *x, Dtype *y);

template<typename Dtype>
void gpu_argmax(const int M, const int D, const int N, const Dtype *x, Dtype *y);
/* self-op end*/

/* vector-scalar start*/

template<typename Dtype>
void gpu_set(const int n, const Dtype alpha, Dtype *x);

template<typename Dtype>
void gpu_add_scalar(const int n, const Dtype *x, const Dtype alpha, Dtype *y);

template<typename Dtype>
void gpu_scale(const int n, const Dtype alpha, Dtype *y);

template<typename Dtype>
void gpu_scale(const int n, const Dtype *x, const Dtype alpha, Dtype *y);

#ifndef gpu_ONLY
template<typename Dtype>
void gpu_scale(const int n, const Dtype alpha, Dtype *x, cudaStream_t str);
#endif

template<typename Dtype>
void gpu_pow_scalar(const int n, const Dtype *a, const Dtype b, Dtype *y);

/* vector-scalar end*/


/* vector-vector start*/
template<typename Dtype>
bool gpu_equal(const int n, const Dtype *x, const Dtype *y);

template<typename Dtype>
void gpu_copy(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void gpu_add(const int n, const Dtype *a, const Dtype *b, Dtype *y);

template<typename Dtype>
void gpu_sub(const int n, const Dtype *a, const Dtype *b, Dtype *y);

template<typename Dtype>
void gpu_mul(const int n, const Dtype *a, const Dtype *b, Dtype *y);

template<typename Dtype>
void gpu_div(const int n, const Dtype *a, const Dtype *b, Dtype *y);

template<typename Dtype>
void gpu_pow(const int n, const Dtype *a, const Dtype *b, Dtype *y);

template<typename Dtype>
void gpu_add_broadcast(const Dtype *a, const Dtype *b,
                       std::vector<int> &shape_a,
                       std::vector<int> &shape_b,
                       Dtype *y);

template<typename Dtype>
void gpu_sub_broadcast(const Dtype *a, const Dtype *b,
                       std::vector<int> &shape_a,
                       std::vector<int> &shape_b,
                       Dtype *y);

template<typename Dtype>
void gpu_mul_broadcast(const Dtype *a, const Dtype *b,
                       std::vector<int> &shape_a,
                       std::vector<int> &shape_b,
                       Dtype *y);

template<typename Dtype>
void gpu_div_broadcast(const Dtype *a, const Dtype *b,
                       std::vector<int> &shape_a,
                       std::vector<int> &shape_b,
                       Dtype *y);
template<typename Dtype>
void gpu_pow_broadcast(const Dtype *a, const Dtype *b,
                       std::vector<int> &shape_a,
                       std::vector<int> &shape_b,
                       Dtype *y);

template<typename Dtype>
void gpu_dot(const int n, const Dtype *x, const Dtype *y, Dtype *out);
template<typename Dtype>
void gpu_stride_dot(int n,
                    const Dtype *x, int incx,
                    const Dtype *y, int incy,
                    Dtype *out);

template<typename Dtype>
void gpu_maximum(const int n,
                 const Dtype *a, const Dtype *b,
                 Dtype *y);

template<typename Dtype>
void gpu_minimum(const int n,
                 const Dtype *a, const Dtype *b,
                 Dtype *y);

/* vector-vector end*/

/* matrix-vector start*/
template<typename Dtype>
void gpu_gemv(const bool TransA, const int M, const int N,
              const Dtype alpha, const Dtype *A, const Dtype *x, const Dtype beta,
              Dtype *y);
/* matrix-vector end*/

/* matrix-matrix start*/
template<typename Dtype>
void gpu_gemm(const bool TransA,
              const bool TransB, const int M, const int N, const int K,
              const Dtype alpha, const Dtype *A, const Dtype *B, const Dtype beta,
              Dtype *C);
template<typename Dtype>
void gpu_axpy(const int n, const Dtype alpha, const Dtype *x,
              Dtype *y);

template<typename Dtype>
void gpu_axpby(const int n, const Dtype alpha, const Dtype *x,
               const Dtype beta, Dtype *y);
/* matrix-matrix end*/

/* random generator start*/
// gpu_rng_uniform with two arguments generates integers in the range
// [0, UINT_MAX].
void gpu_rng_uniform(const int n, unsigned int *r);

template<typename Dtype>
void gpu_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype *r);

template<typename Dtype>
void gpu_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
                      Dtype *r);

template<typename Dtype>
void gpu_rng_bernoulli(const int n, const Dtype p, int *r);
/* random generator end*/

//template<typename Dtype>
//void gpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype *y);

}
#endif //STENSOR_CORE_MATH_BASE_CUDA_HPP_
