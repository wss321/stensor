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

namespace stensor {

/* self op start*/
template<typename Dtype>
void cpu_exp(const uint32_t n,
             const Dtype *a,
             Dtype *y);

template<typename Dtype>
void cpu_log(const uint32_t n,
             const Dtype *a,
             Dtype *y);

template<typename Dtype>
void cpu_abs(const uint32_t n,
             const Dtype *a,
             Dtype *y);

template<typename Dtype>
void cpu_sqrt(const uint32_t n,
              const Dtype *a,
              Dtype *y);

template<typename Dtype>
void cpu_square(const uint32_t n,
                const Dtype *a,
                Dtype *y);

template<typename Dtype>
void cpu_sign(const uint32_t n,
              const Dtype *a,
              Dtype *y);
template<typename Dtype>
void cpu_clamp(const uint32_t n,
               const Dtype min, const Dtype max,
               const Dtype *a,
               Dtype *y);
/* self op end*/

/* vector scalar start*/
template<typename Dtype>
void cpu_set(const uint32_t n,
             const Dtype val,
             Dtype *y);

template<typename Dtype>
void cpu_copy(const uint32_t n,
              const Dtype *a,
              Dtype *y);

template<typename Dtype>
void cpu_add_scalar(const uint32_t n,
                    const Dtype *a, const Dtype val,
                    Dtype *y);

template<typename Dtype>
void cpu_sub_scalar(const uint32_t n,
                    const Dtype *a, const Dtype val,
                    Dtype *y);

template<typename Dtype>
void cpu_scale(const uint32_t n,
               const Dtype *a, const Dtype val,
               Dtype *y);

template<typename Dtype>
void cpu_pow_scalar(const uint32_t n,
                    const Dtype *a, const Dtype val,
                    Dtype *y);

/* vector scalar end*/

/* vector vector start*/
template<typename Dtype>
void cpu_add(const uint32_t n,
             const Dtype *a, const Dtype *b,
             Dtype *y);

template<typename Dtype>
void cpu_sub(const uint32_t n,
             const Dtype *a, const Dtype *b,
             Dtype *y);

template<typename Dtype>
void cpu_mul(const uint32_t n,
             const Dtype *a, const Dtype *b,
             Dtype *y);

template<typename Dtype>
void cpu_div(const uint32_t n,
             const Dtype *a, const Dtype *b,
             Dtype *y);
template<typename Dtype>
void cpu_pow(const uint32_t n,
             const Dtype *a, const Dtype *b,
             Dtype *y);

// Returns the sum of the absolute values of the elements of vector a
template<typename Dtype>
Dtype cpu_asum(uint32_t n, const Dtype *a);

template<typename Dtype>
Dtype cpu_dot(uint32_t n, const Dtype *a, const Dtype *b);

template<typename Dtype>
Dtype cpu_stride_dot(uint32_t n,
                     const Dtype *a, uint32_t incx,
                     const Dtype *y, uint32_t incy);

/* vector vector end*/

/* matrix vector start*/
template<typename Dtype>
void cpu_gemv(const CBLAS_TRANSPOSE TransA,
              const uint32_t M, const uint32_t N,
              const Dtype alpha, const Dtype *A, const Dtype *a,
              const Dtype beta, Dtype *y);

/* matrix vector end*/

/* matrix matrix start*/
template<typename Dtype>
void axpy(const uint32_t N,
          const Dtype alpha, const Dtype *a,
          Dtype *y);

template<typename Dtype>
void cpu_gemm(const CBLAS_TRANSPOSE TransA,
              const CBLAS_TRANSPOSE TransB,
              const uint32_t M, const uint32_t N, const uint32_t K,
              const Dtype alpha, const Dtype *A, const Dtype *B,
              const Dtype beta, Dtype *C);

template<typename Dtype>
void cpu_axpby(const uint32_t N,
               const Dtype alpha, const Dtype *a,
               const Dtype beta, Dtype *y);

/* matrix matrix end*/

/* generator start*/
template<typename Dtype>
void cpu_rng_uniform(uint32_t n,
                     Dtype a, Dtype b,
                     Dtype *r);

template<typename Dtype>
void cpu_rng_gaussian(uint32_t n,
                      Dtype mu, Dtype sigma,
                      Dtype *r);

template<typename Dtype>
void cpu_rng_bernoulli(uint32_t n,
                       Dtype p,
                       uint32_t *r);
/* generator end*/


}
#endif //STENSOR_CORE_MATH_BASE_CPU_HPP_
