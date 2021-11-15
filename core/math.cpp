/**
* Copyright 2021 wss
* Created by wss on 11月,15, 2021
*/
#include "math.hpp"

#define SCINT static_cast<int>

#define DEFINE_VSL_UNARY_FUNC(name, operation) \
  template<typename Dtype> \
  void v##name(const uint32_t n, const Dtype* a, Dtype* y) { \
    CHECK(a); CHECK(y); \
    for (int i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const uint32_t n, const float* a, float* y) { \
    v##name<float>(n, a, y); \
  } \
  inline void vd##name( \
      const uint32_t n, const double* a, double* y) { \
    v##name<double>(n, a, y); \
  }

DEFINE_VSL_UNARY_FUNC(Sqr, y[i] = a[i] * a[i])
DEFINE_VSL_UNARY_FUNC(Sqrt, y[i] = sqrt(a[i]))
DEFINE_VSL_UNARY_FUNC(Exp, y[i] = exp(a[i]))
DEFINE_VSL_UNARY_FUNC(Log, y[i] = log(a[i]))
DEFINE_VSL_UNARY_FUNC(Abs, y[i] = fabs(a[i]))
DEFINE_VSL_UNARY_FUNC(Sign, y[i] = a[i] > 0 ? 1 : -1)

#define DEFINE_VSL_BINARY_FUNC(name, operation) \
  template<typename Dtype> \
  void v##name(const int n, const Dtype* a, const Dtype* b, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(b); CHECK(y); \
    for (int i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int n, const float* a, const float* b, float* y) { \
    v##name<float>(n, a, b, y); \
  } \
  inline void vd##name( \
      const int n, const double* a, const double* b, double* y) { \
    v##name<double>(n, a, b, y); \
  }

class uniform_real;
DEFINE_VSL_BINARY_FUNC(Add, y[i] = a[i] + b[i])
DEFINE_VSL_BINARY_FUNC(Sub, y[i] = a[i] - b[i])
DEFINE_VSL_BINARY_FUNC(Mul, y[i] = a[i] * b[i])
DEFINE_VSL_BINARY_FUNC(Div, y[i] = a[i] / b[i])
DEFINE_VSL_BINARY_FUNC(Pow, y[i] = std::pow(a[i], b[i]))

namespace stensor {
/* self op start*/
template<typename Dtype>
void exp(const uint32_t n,
         const Dtype *a,
         Dtype *y) {
  vExp(n, a, y);
};

template<typename Dtype>
void log(const uint32_t n,
         const Dtype *a,
         Dtype *y) {
  vLog(n, a, y);
};

template<typename Dtype>
void abs(const uint32_t n,
         const Dtype *a,
         Dtype *y) {
  vAbs(n, a, y);
};

template<typename Dtype>
void sqrt(const uint32_t n,
          const Dtype *a,
          Dtype *y) {
  vSqrt(n, a, y);
};

template<typename Dtype>
void square(const uint32_t n,
            const Dtype *a,
            Dtype *y) {
  vSqr(n, a, y);
};

template<typename Dtype>
void sign(const uint32_t n,
          const Dtype *a,
          Dtype *y) {
  vSign(n, a, y);
}

template<typename Dtype>
void clamp(const uint32_t n,
           const Dtype min, const Dtype max,
           const Dtype *a,
           Dtype *y) {
  CHECK_GT(max, min);
  CHECK(a);
  CHECK(y);
  for (int i = 0; i < n; ++i) {
    if (a[i] >= min && a[i] <= max) y[i] = a[i];
    else if (a[i] < min) y[i] = min;
    else if (a[i] > max) y[i] = max;
  }

}
/* self op end*/

/* vector scalar start*/
template<typename Dtype>
void set(const uint32_t n,
         const Dtype val,
         Dtype *y) {
  CHECK(y);
  if (val == 0) {
    std::memset(y, 0, sizeof(Dtype) * n);
    return;
  }
  for (int i = 0; i < n; ++i) {
    y[i] = val;
  }

}
template void set<uint32_t>(const uint32_t n, const uint32_t val, uint32_t *y);
template void set<int>(const uint32_t n, const int val, int *y);
template void set<float>(const uint32_t n, const float val, float *y);
template void set<double>(const uint32_t n, const double val, double *y);

template<typename Dtype>
void cpu_copy(const uint32_t n, const Dtype *X, Dtype *Y) {
  CHECK(X);
  CHECK(Y);
  if (X != Y) std::memcpy(Y, X, sizeof(Dtype) * n);
}
template void cpu_copy<uint32_t>(const uint32_t n, const uint32_t *X, uint32_t *Y);
template void cpu_copy<int>(const uint32_t n, const int *X, int *Y);
template void cpu_copy<float>(const uint32_t n, const float *X, float *Y);
template void cpu_copy<double>(const uint32_t n, const double *X, double *Y);

template<typename Dtype>
void add(const uint32_t n,
         const Dtype *a, const Dtype val,
         Dtype *y) {
  CHECK(a);
  CHECK(y);
  for (int i = 0; i < n; ++i) {
    y[i] = a[i] + val;
  }
}
template void add<uint32_t>(const uint32_t n, const uint32_t *a, const uint32_t val, uint32_t *y);
template void add<int>(const uint32_t n, const int *a, const int val, int *y);
template void add<float>(const uint32_t n, const float *a, const float val, float *y);
template void add<double>(const uint32_t n, const double *a, const double val, double *y);

template<typename Dtype>
void sub(const uint32_t n,
         const Dtype *a, const Dtype val,
         Dtype *y) {
  CHECK(a);
  CHECK(y);
  for (int i = 0; i < n; ++i) {
    y[i] = a[i] - val;
  }
}
template void sub<uint32_t>(const uint32_t n, const uint32_t *a, const uint32_t val, uint32_t *y);
template void sub<int>(const uint32_t n, const int *a, const int val, int *y);
template void sub<float>(const uint32_t n, const float *a, const float val, float *y);
template void sub<double>(const uint32_t n, const double *a, const double val, double *y);

template<typename Dtype>
void scale(const uint32_t n,
           const Dtype *a, const Dtype val,
           Dtype *y) {
  CHECK(a);
  CHECK(y);
  for (int i = 0; i < n; ++i) {
    y[i] = a[i] * val;
  }
}
template void scale<uint32_t>(const uint32_t n, const uint32_t *a, const uint32_t val, uint32_t *y);
template void scale<int>(const uint32_t n, const int *a, const int val, int *y);
template void scale<float>(const uint32_t n, const float *a, const float val, float *y);
template void scale<double>(const uint32_t n, const double *a, const double val, double *y);

template<typename Dtype>
void pow(const uint32_t n,
         const Dtype *a, const Dtype val,
         Dtype *y) {
  CHECK(a);
  CHECK(y);
  for (int i = 0; i < n; ++i) {
    y[i] = std::pow(a[i], val);
  }
}
template void pow<uint32_t>(const uint32_t n, const uint32_t *a, const uint32_t val, uint32_t *y);
template void pow<int>(const uint32_t n, const int *a, const int val, int *y);
template void pow<float>(const uint32_t n, const float *a, const float val, float *y);
template void pow<double>(const uint32_t n, const double *a, const double val, double *y);


/* vector scalar end*/

/* vector vector start*/
template<typename Dtype>
void add(const uint32_t n,
         const Dtype *a, const Dtype *b,
         Dtype *y) {
  vAdd(n, a, b, y);
}

template void add<uint32_t>(const uint32_t n, const uint32_t *a, const uint32_t *b, uint32_t *y);
template void add<int>(const uint32_t n, const int *a, const int *b, int *y);
template void add<float>(const uint32_t n, const float *a, const float *b, float *y);
template void add<double>(const uint32_t n, const double *a, const double *b, double *y);

template<typename Dtype>
void sub(const uint32_t n,
         const Dtype *a, const Dtype *b,
         Dtype *y) {
  vSub(n, a, b, y);
}
template void sub<uint32_t>(const uint32_t n, const uint32_t *a, const uint32_t *b, uint32_t *y);
template void sub<int>(const uint32_t n, const int *a, const int *b, int *y);
template void sub<float>(const uint32_t n, const float *a, const float *b, float *y);
template void sub<double>(const uint32_t n, const double *a, const double *b, double *y);

template<typename Dtype>
void mul(const uint32_t n,
         const Dtype *a, const Dtype *b,
         Dtype *y) {
  vMul(n, a, b, y);
}
template void mul<uint32_t>(const uint32_t n, const uint32_t *a, const uint32_t *b, uint32_t *y);
template void mul<int>(const uint32_t n, const int *a, const int *b, int *y);
template void mul<float>(const uint32_t n, const float *a, const float *b, float *y);
template void mul<double>(const uint32_t n, const double *a, const double *b, double *y);

template<typename Dtype>
void div(const uint32_t n,
         const Dtype *a, const Dtype *b,
         Dtype *y) {
  vDiv(n, a, b, y);
}
template void div<uint32_t>(const uint32_t n, const uint32_t *a, const uint32_t *b, uint32_t *y);
template void div<int>(const uint32_t n, const int *a, const int *b, int *y);
template void div<float>(const uint32_t n, const float *a, const float *b, float *y);
template void div<double>(const uint32_t n, const double *a, const double *b, double *y);

template<typename Dtype>
void pow(const uint32_t n,
         const Dtype *a, const Dtype *b,
         Dtype *y) {
  vPow(n, a, b, y);
}
template void pow<uint32_t>(const uint32_t n, const uint32_t *a, const uint32_t *b, uint32_t *y);
template void pow<int>(const uint32_t n, const int *a, const int *b, int *y);
template void pow<float>(const uint32_t n, const float *a, const float *b, float *y);
template void pow<double>(const uint32_t n, const double *a, const double *b, double *y);

// Returns the sum of the absolute values of the elements of vector x
template<>
float cpu_asum<float>(uint32_t n, const float *x) {
  return cblas_sasum(SCINT(n), x, 1);
}

template<>
float cpu_dot<float>(uint32_t n, const float *x, const float *y) {
  return cblas_sasum(SCINT(n), x, 1);
}

template<>
float cpu_stride_dot<float>(const uint32_t n, const float *x, const uint32_t incx,
                            const float *y, const uint32_t incy) {
  return cblas_sdot(SCINT(n), x, SCINT(incx), y, SCINT(incy));
}

/* vector vector end*/

/* matrix vector start*/
template<>
void cpu_gemv<float>(const CBLAS_TRANSPOSE TransA,
                     const uint32_t M, const uint32_t N,
                     const float alpha, const float *A, const float *a,
                     const float beta, float *y) {
  cblas_sgemv(CblasRowMajor, TransA, SCINT(M), SCINT(N),
              alpha, A, SCINT(N), a, 1, beta, y, 1);
}
template<>
void cpu_gemv<double>(const CBLAS_TRANSPOSE TransA,
                      const uint32_t M, const uint32_t N,
                      const double alpha, const double *A, const double *a,
                      const double beta, double *y) {
  cblas_dgemv(CblasRowMajor, TransA, SCINT(M), SCINT(N),
              alpha, A, SCINT(N), a, 1, beta, y, 1);
}

/* matrix vector end*/

/* matrix matrix start*/
template<>
void axpy<float>(const uint32_t N,
                 const float alpha, const float *a,
                 float *y) {
  cblas_saxpy(SCINT(N), alpha, a, 1, y, 1);
}
template<>
void axpy<double>(const uint32_t N,
                  const double alpha, const double *a,
                  double *y) {
  cblas_daxpy(SCINT(N), alpha, a, 1, y, 1);
}

template<>
void cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
                     const CBLAS_TRANSPOSE TransB,
                     const uint32_t M, const uint32_t N, const uint32_t K,
                     const float alpha, const float *A, const float *B,
                     const float beta, float *C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, SCINT(M),
              SCINT(N), SCINT(K), alpha, A, lda, B, ldb,
              beta, C, SCINT(N));
}
template<>
void cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
                      const CBLAS_TRANSPOSE TransB,
                      const uint32_t M, const uint32_t N, const uint32_t K,
                      const double alpha, const double *A, const double *B,
                      const double beta, double *C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, SCINT(M),
              SCINT(N), SCINT(K), alpha, A, lda, B, ldb,
              beta, C, SCINT(N));
}

template<>
void cpu_axpby<float>(const uint32_t N,
                      const float alpha, const float *a,
                      const float beta, float *y) {
  cblas_saxpby(SCINT(N), alpha, a, 1, beta, y, 1);
}
template<>
void cpu_axpby<double>(const uint32_t N,
                       const double alpha, const double *a,
                       const double beta, double *y) {
  cblas_daxpby(SCINT(N), alpha, a, 1, beta, y, 1);
}

/* matrix matrix end*/

/* generator start*/
template<typename Dtype>
void rng_uniform(uint32_t n,
                 Dtype a, Dtype b,
                 Dtype *r) {
  CHECK_LE(n, INT32_MAX);
  CHECK(r);
  CHECK_LE(a, b);

  std::random_device rd;
  std::mt19937 gen{rd()};

  // values near the mean are the most likely
  // standard deviation affects the dispersion of generated values from the mean
  std::uniform_real_distribution<Dtype> dis(a, b);
  for (uint32_t i = 0; i < n; ++i) {
    r[i] = dis(gen);
  }
}
template void rng_uniform<float>(uint32_t n, float a, float b, float *r);
template void rng_uniform<double>(uint32_t n, double a, double b, double *r);

template<typename Dtype>
void rng_gaussian(uint32_t n,
                  Dtype mu, Dtype sigma,
                  Dtype *r) {
  CHECK_LE(n, INT32_MAX);
  CHECK_LE(sigma, INT32_MAX);
  CHECK(r);
  std::normal_distribution<Dtype> random_distribution(mu, sigma);

  std::random_device rd{};
  std::mt19937 gen{rd()};
  for (uint32_t i = 0; i < n; ++i) {
    r[i] = random_distribution(gen);
  }
}
template void rng_gaussian<float>(uint32_t n, float mu, float sigma, float *r);
template void rng_gaussian<double>(uint32_t n, double mu, double sigma, double *r);

template<typename Dtype>
void rng_bernoulli(uint32_t n,
                   Dtype p,
                   uint32_t *r) {
  CHECK_LE(n, INT32_MAX);
  CHECK(r);
  CHECK_LE(p, 1);
  std::bernoulli_distribution random_distribution(p);

  std::random_device rd{};
  std::mt19937 gen{rd()};
  for (uint32_t i = 0; i < n; ++i) {
    r[i] = random_distribution(gen);
  }
}
template void rng_bernoulli<float>(uint32_t n, float a, uint32_t *r);
template void rng_bernoulli<double>(uint32_t n, double a, uint32_t *r);

/* generator end*/


/* math of Tensor */

/* math of Tensor end */

}//namespace stensor