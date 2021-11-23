/**
* Copyright 2021 wss
* Created by wss on 11月,15, 2021
*/
#include "math_base_cpu.hpp"
namespace stensor {

#define INITIAL_UNARY_FUNC_PP(name) \
  template void name<int>(const int n, const int *a, int *y); \
  template void name<float>(const int n, const float *a, float *y); \
  template void name<double>(const int n, const double *a, double *y);

#define INITIAL_UNARY_FUNC_SP(name) \
  template void name<int>(const int n, const int val, int *y); \
  template void name<float>(const int n, const float val, float *y); \
  template void name<double>(const int n, const double val, double *y);

#define INITIAL_BINARY_FUNC_PSP(name) \
  template void name<int>(const int n, const int *a, const int val, int *y); \
  template void name<float>(const int n, const float *a, const float val, float *y); \
  template void name<double>(const int n, const double *a, const double val, double *y);

/* self op start*/

#define IMPLEMENT_CPU_UNARY_FUNC(name, op_expression)\
template<typename Dtype>\
void cpu_##name(const int n,\
             const Dtype *x,\
             Dtype *y) {\
  CHECK_GT(n, 0); CHECK(x); CHECK(y);\
  for (int index = 0; index < n; ++index) { op_expression; }\
}\
template void cpu_##name<float>(const int n, const float *x, float *y);\
template void cpu_##name<double>(const int n, const double *x, double *y);

IMPLEMENT_CPU_UNARY_FUNC(exp, y[index] = exp(x[index]));
IMPLEMENT_CPU_UNARY_FUNC(log, y[index] = log(x[index]));
IMPLEMENT_CPU_UNARY_FUNC(abs, y[index] = abs(x[index]));
IMPLEMENT_CPU_UNARY_FUNC(sqrt, y[index] = sqrt(x[index]));
IMPLEMENT_CPU_UNARY_FUNC(square, y[index] = x[index] * x[index]);
IMPLEMENT_CPU_UNARY_FUNC(sign, y[index] = x[index] > Dtype(0.0) ? Dtype(1.0) : -Dtype(1.0));
IMPLEMENT_CPU_UNARY_FUNC(sigmoid, y[index] = Dtype(1.0) / (Dtype(1.0) + exp(-x[index])))
//IMPLEMENT_CPU_UNARY_FUNC(tanh, Dtype e_2x = exp(Dtype(2.0) * x[index]);y[index] = (e_2x-Dtype(1.0))/(e_2x+Dtype(1.0)));
IMPLEMENT_CPU_UNARY_FUNC(tanh, y[index] = tanh(x[index]));
IMPLEMENT_CPU_UNARY_FUNC(relu, y[index] = x[index] > Dtype(0.0) ? x[index] : Dtype(0.0));
IMPLEMENT_CPU_UNARY_FUNC(elu, y[index] = x[index] > Dtype(0.0) ? x[index] : exp(x[index]) - Dtype(1.0));
IMPLEMENT_CPU_UNARY_FUNC(gelu, y[index] = Dtype(0.5) * x[index] * (Dtype(1.0) + erf(x[index] / sqrt(Dtype(2.0)))));
IMPLEMENT_CPU_UNARY_FUNC(leakyrelu, y[index] = x[index] > 0 ? x[index] : Dtype(0.2) * x[index]);

template<typename Dtype>
void cpu_clamp(const int n,
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
template void cpu_clamp<int>(const int n, const int min, const int max, const int *a, int *y);
template void cpu_clamp<float>(const int n, const float min, const float max, const float *a, float *y);
template void cpu_clamp<double>(const int n, const double min, const double max, const double *a, double *y);

template<typename Dtype>
void cpu_reduce_sum(const int M, const int D, const int N, const Dtype *x, Dtype beta, Dtype *y) {
  const Dtype *in_data = x;
  Dtype *out_data = y;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      Dtype sum = 0;
      for (int d = 0; d < D; ++d) {
        sum += in_data[d * N + n];
      }
      if (beta == 0) *out_data = sum;
      else *out_data = sum + beta * (*out_data);
      out_data++;
    }
    in_data += D * N;
  }
}

template void cpu_reduce_sum<int>(const int M, const int D, const int N, const int *x, int beta, int *y);
template void cpu_reduce_sum<float>(const int M, const int D, const int N, const float *x, float beta, float *y);
template void cpu_reduce_sum<double>(const int M, const int D, const int N, const double *x, double beta, double *y);

template<typename Dtype>
void cpu_softmax(const int M, const int D, const int N, const Dtype *x, Dtype beta, Dtype *y) {
  const Dtype *in_data = x;
  Dtype *out_data = y;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      // calc denominator
      Dtype denominator = 0;
      for (int d = 0; d < D; ++d) {
        denominator += exp(in_data[d * N + n]);
      }

      if (beta == 0) {
        for (int d = 0; d < D; ++d) {
          out_data[d * N + n] = exp(in_data[d * N + n]) / denominator;
        }
      } else {
        for (int d = 0; d < D; ++d) {
          out_data[d * N + n] = beta * (*out_data) + exp(in_data[d * N + n]) / denominator;
        }
      }
    }
    in_data += D * N;
    out_data += D * N;
  }
}

template void cpu_softmax<int>(const int M, const int D, const int N, const int *x, int beta, int *y);
template void cpu_softmax<float>(const int M, const int D, const int N, const float *x, float beta, float *y);
template void cpu_softmax<double>(const int M, const int D, const int N, const double *x, double beta, double *y);

template<typename Dtype>
void cpu_reduce_mean(const int M, const int D, const int N, const Dtype *x, Dtype beta, Dtype *y) {
  const Dtype *in_data = x;
  Dtype *out_data = y;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      Dtype sum = 0;
      for (int d = 0; d < D; ++d) {
        sum += in_data[d * N + n];
      }
      if (beta == 0) *out_data = sum / D;
      else *out_data = sum / D + beta * (*out_data);
      out_data++;
    }
    in_data += D * N;
  }
}
template void cpu_reduce_mean<int>(const int M, const int D, const int N, const int *x, int beta, int *y);
template void cpu_reduce_mean<float>(const int M, const int D, const int N, const float *x, float beta, float *y);
template void cpu_reduce_mean<double>(const int M, const int D, const int N, const double *x, double beta, double *y);

template<typename Dtype>
void cpu_reduce_asum(const int M, const int D, const int N, const Dtype *x, Dtype beta, Dtype *y) {
  const Dtype *in_data = x;
  Dtype *out_data = y;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      Dtype sum = 0;
      for (int d = 0; d < D; ++d) {
        sum += abs(in_data[d * N + n]);
      }
      if (beta == 0) *out_data = sum;
      else *out_data = sum + beta * (*out_data);
      out_data++;
    }
    in_data += D * N;
  }
}
template void cpu_reduce_asum<int>(const int M, const int D, const int N, const int *x, int beta, int *y);
template void cpu_reduce_asum<float>(const int M, const int D, const int N, const float *x, float beta, float *y);
template void cpu_reduce_asum<double>(const int M, const int D, const int N, const double *x, double beta, double *y);


/* self op end*/

/* vector scalar start*/
template<typename Dtype>
bool cpu_equal(const int n,
               const Dtype *a,
               const Dtype *b) {
  CHECK(a);
  CHECK(b);
  for (int i = 0; i < n; ++i)
    if (a[i] != b[i]) return false;
  return true;
}
template bool cpu_equal<int>(const int n, const int *a, const int *b);
template bool cpu_equal<float>(const int n, const float *a, const float *b);
template bool cpu_equal<double>(const int n, const double *a, const double *b);

template<typename Dtype>
void cpu_set(const int n,
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
INITIAL_UNARY_FUNC_SP(cpu_set);
template<typename Dtype>
void cpu_copy(const int n, const Dtype *x, Dtype *y) {
  CHECK(x);
  CHECK(y);
  if (x != y) std::memcpy(y, x, sizeof(Dtype) * n);
  //cblas_ccopy(static_cast<int>(n), x, 1, y, 1);
}
INITIAL_UNARY_FUNC_PP(cpu_copy);

template<typename Dtype>
void cpu_add_scalar(const int n,
                    const Dtype *a, const Dtype val,
                    Dtype *y) {
  CHECK(a);
  CHECK(y);
  for (int i = 0; i < n; ++i) {
    y[i] = a[i] + val;
  }
}
INITIAL_BINARY_FUNC_PSP(cpu_add_scalar);

template<typename Dtype>
void cpu_sub_scalar(const int n,
                    const Dtype *a, const Dtype val,
                    Dtype *y) {
  CHECK(a);
  CHECK(y);
  for (int i = 0; i < n; ++i) {
    y[i] = a[i] - val;
  }
}
INITIAL_BINARY_FUNC_PSP(cpu_sub_scalar);

template<typename Dtype>
void cpu_scale(const int n,
               const Dtype *a, const Dtype val,
               Dtype *y) {
  CHECK(a);
  CHECK(y);
  for (int i = 0; i < n; ++i) {
    y[i] = a[i] * val;
  }
}
INITIAL_BINARY_FUNC_PSP(cpu_scale);

template<typename Dtype>
void cpu_pow_scalar(const int n,
                    const Dtype *a, const Dtype val,
                    Dtype *y) {
  CHECK(a);
  CHECK(y);
  for (int i = 0; i < n; ++i) {
    y[i] = std::pow(a[i], val);
  }
}
INITIAL_BINARY_FUNC_PSP(cpu_pow_scalar);

/* vector scalar end*/

/* vector vector start*/

#define IMPLEMENT_CPU_BINARY_FUNC(name, op_expression) \
  template<typename Dtype> \
  void cpu_##name(const int n, const Dtype* a, const Dtype* b, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(b); CHECK(y); \
    for (int index = 0; index < n; ++index) { op_expression; } \
  }                                             \
template void cpu_##name<int>(const int n, const int *a, const int *b, int *y);\
template void cpu_##name<float>(const int n, const float *a, const float *b, float *y);\
template void cpu_##name<double>(const int n, const double *a, const double *b, double *y)

IMPLEMENT_CPU_BINARY_FUNC(add, y[index] = a[index] + b[index]);
IMPLEMENT_CPU_BINARY_FUNC(sub, y[index] = a[index] - b[index]);
IMPLEMENT_CPU_BINARY_FUNC(mul, y[index] = a[index] * b[index]);
IMPLEMENT_CPU_BINARY_FUNC(div, y[index] = a[index] / b[index]);
IMPLEMENT_CPU_BINARY_FUNC(pow, y[index] = pow(a[index], b[index]));
IMPLEMENT_CPU_BINARY_FUNC(maximum, y[index] = std::max(a[index], b[index]));
IMPLEMENT_CPU_BINARY_FUNC(minimum, y[index] = std::min(a[index], b[index]));

#define BROADCAST_INDEX(index, n, num_axis, indices_in_result, shape_a, shape_b, shape_y, index_a, index_b) \
  indices_in_result[num_axis - 1] = index % shape_y[num_axis - 1]; \
  int div = 1; \
  for (int i = num_axis - 2; i >= 0; --i) { \
    div *=  shape_y[i + 1]; \
    indices_in_result[i] = (index / div)% shape_y[i]; \
  } \
  int index_a = 0; \
  int index_b = 0; \
  for (int i = 0; i < num_axis; ++i) { \
    int ma = indices_in_result[i] > shape_a[i] ? shape_a[i] : indices_in_result[i];\
    int mb = indices_in_result[i] > shape_b[i] ? shape_b[i] : indices_in_result[i];\
    index_a *= shape_a[i]; \
    index_b *= shape_b[i]; \
    if (shape_a[i]!=1)  index_a += ma; \
    if (shape_b[i]!=1)  index_b += mb; \
  }

#define IMPLEMENT_BINARY_BROADCAST_CPU_FUNC(name, type, op_expression) \
template<>\
void cpu_##name<type>(const type *a, const type *b,\
                       std::vector<int>& shape_a,\
                       std::vector<int>& shape_b,\
                       type *o){\
  const std::vector<int> shape_out = stensor::broadcast(shape_a, shape_b);\
  int size = 1;\
  type * y = o;                                                        \
  for (int i = 0; i < shape_out.size(); ++i)    size*= shape_out[i];   \
  int indices_in_result[MAX_AXES]{0};                                     \
  for (int index = 0; index < size; ++index) {\
          BROADCAST_INDEX(index, size, shape_out.size(), indices_in_result, shape_a, shape_b, shape_out, index_a, index_b);\
          op_expression;\
        }\
}

IMPLEMENT_BINARY_BROADCAST_CPU_FUNC(add_broadcast, float, *y = a[index_a] + b[index_b];y++;);
IMPLEMENT_BINARY_BROADCAST_CPU_FUNC(add_broadcast, double, *y = a[index_a] + b[index_b];y++;);

IMPLEMENT_BINARY_BROADCAST_CPU_FUNC(sub_broadcast, float, *y = a[index_a] - b[index_b];y++;);
IMPLEMENT_BINARY_BROADCAST_CPU_FUNC(sub_broadcast, double, *y = a[index_a] - b[index_b];y++;);

IMPLEMENT_BINARY_BROADCAST_CPU_FUNC(mul_broadcast, float, *y = a[index_a] * b[index_b];y++;);
IMPLEMENT_BINARY_BROADCAST_CPU_FUNC(mul_broadcast, double, *y = a[index_a] * b[index_b];y++;);

IMPLEMENT_BINARY_BROADCAST_CPU_FUNC(div_broadcast, float, *y = a[index_a] / b[index_b];y++;);
IMPLEMENT_BINARY_BROADCAST_CPU_FUNC(div_broadcast, double, *y = a[index_a] / b[index_b];y++;);

IMPLEMENT_BINARY_BROADCAST_CPU_FUNC(pow_broadcast, float, *y = std::pow(a[index_a], b[index_b]);y++;);
IMPLEMENT_BINARY_BROADCAST_CPU_FUNC(pow_broadcast, double, *y = std::pow(a[index_a], b[index_b]);y++;);

template<>
float cpu_asum<float>(int n, const float *x) {
  return cblas_sasum(n, x, 1);
}

template<>
float cpu_dot<float>(int n, const float *x, const float *y) {
  return cblas_sdot(n, x, 1, y, 1);
}

template<>
float cpu_stride_dot<float>(const int n, const float *x, const int incx,
                            const float *y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

/* vector vector end*/

/* matrix vector start*/
template<>
void cpu_gemv<float>(const bool TransA,
                     const int M, const int N,
                     const float alpha, const float *A, const float *a,
                     const float beta, float *y) {
  CBLAS_TRANSPOSE transA = TransA ? CblasTrans : CblasNoTrans;
  cblas_sgemv(CblasRowMajor, transA, M, N,
              alpha, A, N, a, 1, beta, y, 1);
}
template<>
void cpu_gemv<double>(const bool TransA,
                      const int M, const int N,
                      const double alpha, const double *A, const double *a,
                      const double beta, double *y) {
  CBLAS_TRANSPOSE transA = TransA ? CblasTrans : CblasNoTrans;
  cblas_dgemv(CblasRowMajor, transA, M, N,
              alpha, A, N, a, 1, beta, y, 1);
}

/* matrix vector end*/

/* matrix matrix start*/
template<>
void cpu_axpy<float>(const int N,
                     const float alpha, const float *a,
                     float *y) {
  cblas_saxpy(N, alpha, a, 1, y, 1);
}
template<>
void cpu_axpy<double>(const int N,
                      const double alpha, const double *a,
                      double *y) {
  cblas_daxpy(N, alpha, a, 1, y, 1);
}

template<>
void cpu_gemm<float>(const bool TransA,
                     const bool TransB,
                     const int M, const int N, const int K,
                     const float alpha, const float *A, const float *B,
                     const float beta, float *C) {
  CBLAS_TRANSPOSE transA = TransA ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transB = TransB ? CblasTrans : CblasNoTrans;
  int lda = !TransA ? K : M;
  int ldb = !TransB ? N : K;
  cblas_sgemm(CblasRowMajor, transA, transB, M,
              N, K, alpha, A, lda, B, ldb,
              beta, C, N);
}
template<>
void cpu_gemm<double>(const bool TransA,
                      const bool TransB,
                      const int M, const int N, const int K,
                      const double alpha, const double *A, const double *B,
                      const double beta, double *C) {
  CBLAS_TRANSPOSE transA = TransA ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transB = TransB ? CblasTrans : CblasNoTrans;
  int lda = !TransA ? K : M;
  int ldb = !TransB ? N : K;
  cblas_dgemm(CblasRowMajor, transA, transB, M,
              N, K, alpha, A, lda, B, ldb,
              beta, C, N);
}

template<>
void cpu_axpby<float>(const int N,
                      const float alpha, const float *a,
                      const float beta, float *y) {
  cblas_saxpby(N, alpha, a, 1, beta, y, 1);
}
template<>
void cpu_axpby<double>(const int N,
                       const double alpha, const double *a,
                       const double beta, double *y) {
  cblas_daxpby(N, alpha, a, 1, beta, y, 1);
}

/* matrix matrix end*/

/* random generator start*/
//template<typename Dtype>
//void cpu_rng_uniform(int n,
//                     Dtype a, Dtype b,
//                     Dtype *r) {
//  CHECK_LE(n, INT32_MAX);
//  CHECK(r);
//  CHECK_LE(a, b);
//
//  std::random_device rd;
//  std::mt19937 gen{rd()};
//
//  // values near the mean are the most likely
//  // standard deviation affects the dispersion of generated values from the mean
//  std::uniform_real_distribution<Dtype> dis(a, b);
//  for (int i = 0; i < n; ++i) {
//    r[i] = dis(gen);
//  }
//}

template<typename Dtype>
void cpu_rng_uniform(int n,
                     Dtype a, Dtype b,
                     Dtype *r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, std::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max()));
  boost::variate_generator<stensor::rng_t *, boost::uniform_real<Dtype> >
      variate_generator(global_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template void cpu_rng_uniform<float>(int n, float a, float b, float *r);
template void cpu_rng_uniform<double>(int n, double a, double b, double *r);

//template<typename Dtype>
//void cpu_rng_gaussian(int n,
//                      Dtype mu, Dtype sigma,
//                      Dtype *r) {
//  CHECK_LE(n, INT32_MAX);
//  CHECK_LE(sigma, INT32_MAX);
//  CHECK(r);
//  std::normal_distribution<Dtype> random_distribution(mu, sigma);
//
//  std::random_device rd{};
//  std::mt19937 gen{rd()};
//  for (int i = 0; i < n; ++i) {
//    r[i] = random_distribution(gen);
//  }
//}

template<typename Dtype>
void cpu_rng_gaussian(int n,
                      Dtype mu, Dtype sigma,
                      Dtype *r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(mu, sigma);
  boost::variate_generator<stensor::rng_t *, boost::normal_distribution<Dtype> >
      variate_generator(global_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template void cpu_rng_gaussian<float>(int n, float mu, float sigma, float *r);
template void cpu_rng_gaussian<double>(int n, double mu, double sigma, double *r);

//template<typename Dtype>
//void cpu_rng_bernoulli(int n,
//                       Dtype p,
//                       int *r) {
//  CHECK_LE(n, INT32_MAX);
//  CHECK(r);
//  CHECK_LE(p, 1);
//  std::bernoulli_distribution random_distribution(p);
//
//  std::random_device rd{};
//  std::mt19937 gen{rd()};
//  for (int i = 0; i < n; ++i) {
//    r[i] = random_distribution(gen);
//  }
//}

template<typename Dtype>
void cpu_rng_bernoulli(int n, Dtype p, int *r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<stensor::rng_t *, boost::bernoulli_distribution<Dtype> >
      variate_generator(global_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template void cpu_rng_bernoulli<float>(int n, float a, int *r);
template void cpu_rng_bernoulli<double>(int n, double a, int *r);

/* generator end*/

}//namespace stensor