/**
* Copyright 2021 wss
* Created by wss on 11月,16, 2021
*/
#ifndef STENSOR_CORE_MATH_BASE_CUDA_HPP_
#define STENSOR_CORE_MATH_BASE_CUDA_HPP_
#include "common.hpp"
#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cudnn.h>
#include <cstdint>
#include <cmath>
#include <cblas-atlas.h>
#include "memory_op.hpp"
#include "stensor_random.hpp"

namespace stensor {

template<typename Dtype>
void gpu_gemm(const CBLAS_TRANSPOSE TransA,
              const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
              const Dtype alpha, const Dtype *A, const Dtype *B, const Dtype beta,
              Dtype *C);

template<typename Dtype>
void gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
              const Dtype alpha, const Dtype *A, const Dtype *x, const Dtype beta,
              Dtype *y);

template<typename Dtype>
void gpu_axpy(const int N, const Dtype alpha, const Dtype *X,
              Dtype *Y);

template<typename Dtype>
void gpu_axpby(const int N, const Dtype alpha, const Dtype *X,
               const Dtype beta, Dtype *Y);

void gpu_copy(const size_t N, const void *X, void *Y);

template<typename Dtype>
void gpu_set(const int N, const Dtype alpha, Dtype *X);

template<typename Dtype>
void gpu_add_scalar(const int N, const Dtype *X, const Dtype alpha, Dtype *Y);

template<typename Dtype>
void gpu_scale(const int N, const Dtype alpha, Dtype *X);

template<typename Dtype>
void gpu_scale(const int N, const Dtype *X, const Dtype alpha, Dtype *Y);

#ifndef CPU_ONLY
template<typename Dtype>
void gpu_scale(const int N, const Dtype alpha, Dtype *X, cudaStream_t str);
#endif

template<typename Dtype>
void gpu_add(const int N, const Dtype *a, const Dtype *b, Dtype *y);

//TODO:Broadcast add
template<typename Dtype>
void gpu_add_broadcast(const Dtype *a, const Dtype *b,
                       std::vector<uint32_t>& shape_a,
                       std::vector<uint32_t>& shape_b,
                       Dtype *y);

template<typename Dtype>
void gpu_sub(const int N, const Dtype *a, const Dtype *b, Dtype *y);
template<typename Dtype>
void gpu_sub_broadcast(const Dtype *a, const Dtype *b,
                       std::vector<uint32_t>& shape_a,
                       std::vector<uint32_t>& shape_b,
                       Dtype *y);

template<typename Dtype>
void gpu_mul(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template<typename Dtype>
void gpu_mul_broadcast(const Dtype *a, const Dtype *b,
                       std::vector<uint32_t>& shape_a,
                       std::vector<uint32_t>& shape_b,
                       Dtype *y);

template<typename Dtype>
void gpu_div(const int N, const Dtype *a, const Dtype *b, Dtype *y);
template<typename Dtype>
void gpu_div_broadcast(const Dtype *a, const Dtype *b,
                       std::vector<uint32_t>& shape_a,
                       std::vector<uint32_t>& shape_b,
                       Dtype *y);

template<typename Dtype>
void gpu_abs(const int n, const Dtype *a, Dtype *y);

template<typename Dtype>
void gpu_exp(const int n, const Dtype *a, Dtype *y);

template<typename Dtype>
void gpu_log(const int n, const Dtype *a, Dtype *y);

template<typename Dtype>
void gpu_pow_scalar(const int n, const Dtype *a, const Dtype b, Dtype *y);

template<typename Dtype>
void gpu_sqrt(const int n, const Dtype *a, Dtype *y);

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

template<typename Dtype>
void gpu_dot(const int n, const Dtype *x, const Dtype *y, Dtype *out);

template<typename Dtype>
void gpu_asum(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void gpu_sign(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void gpu_sgnbit(const int n, const Dtype *x, Dtype *y);

template<typename Dtype>
void gpu_fabs(const int n, const Dtype *x, Dtype *y);

//template<typename Dtype>
//void gpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype *y);

}
#endif //STENSOR_CORE_MATH_BASE_CUDA_HPP_
