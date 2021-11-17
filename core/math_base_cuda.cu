/**
* Copyright 2021 wss
* Created by wss on 11月,16, 2021
*/

#include "math_base_cuda.hpp"

namespace stensor {

#define DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(name, operation) \
template<typename Dtype> \
__global__ void name##_kernel(const int n, const Dtype* x, Dtype* y) { \
  CUDA_KERNEL_LOOP(index, n) { \
    operation; \
  } \
} \
template <> \
void gpu_##name<float>(const int n, const float* x, float* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<float><<<GET_BLOCKS(n), CUDA_NUM_THREADS>>>( \
      n, x, y); \
} \
template <> \
void gpu_##name<double>(const int n, const double* x, double* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<double><<<GET_BLOCKS(n), CUDA_NUM_THREADS>>>( \
      n, x, y); \
}

template<>
void gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
                     const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                     const float alpha, const float *A, const float *B, const float beta,
                     float *C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Config::cublas_handle(), cuTransB, cuTransA,
                           N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template<>
void gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
                      const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                      const double alpha, const double *A, const double *B, const double beta,
                      double *C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Config::cublas_handle(), cuTransB, cuTransA,
                           N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template<>
void gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
                     const int N, const float alpha, const float *A, const float *x,
                     const float beta, float *y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Config::cublas_handle(), cuTransA, N, M, &alpha,
                           A, N, x, 1, &beta, y, 1));
}

template<>
void gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
                      const int N, const double alpha, const double *A, const double *x,
                      const double beta, double *y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Config::cublas_handle(), cuTransA, N, M, &alpha,
                           A, N, x, 1, &beta, y, 1));
}

template<>
void gpu_axpy<float>(const int N, const float alpha, const float *X,
                     float *Y) {
  CUBLAS_CHECK(cublasSaxpy(Config::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template<>
void gpu_axpy<double>(const int N, const double alpha, const double *X,
                      double *Y) {
  CUBLAS_CHECK(cublasDaxpy(Config::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void gpu_copy(const size_t N, const void *X, void *Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(stensor/alt_fn)
  }
}

template<>
void gpu_scale<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Config::cublas_handle(), N, &alpha, X, 1));
}

template<>
void gpu_scale<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Config::cublas_handle(), N, &alpha, X, 1));
}

template<typename Dtype>
__global__ void scale_kernel(const int n, const Dtype *x, const Dtype alpha, Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = x[index] * alpha;
  }
}

template<>
void gpu_scale(const int N, const float *X, const float alpha, float *Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  if (X == Y)
    gpu_scale<float>(N, alpha, Y);
  else
    scale_kernel < float ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, X, alpha, Y);
}

template<>
void gpu_scale(const int N, const double *X, const double alpha, double *Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  if (X == Y)
    gpu_scale<double>(N, alpha, Y);
  else
    scale_kernel < double ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, X, alpha, Y);
}

template<>
void gpu_scale<float>(const int N, const float alpha, float *X,
                      cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Config::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Config::cublas_handle(), str));
  CUBLAS_CHECK(cublasSscal(Config::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Config::cublas_handle(), initial_stream));
}

template<>
void gpu_scale<double>(const int N, const double alpha, double *X,
                       cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Config::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Config::cublas_handle(), str));
  CUBLAS_CHECK(cublasDscal(Config::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Config::cublas_handle(), initial_stream));
}

template<>
void gpu_axpby<float>(const int N, const float alpha, const float *X,
                      const float beta, float *Y) {
  gpu_scale<float>(N, beta, Y);
  gpu_axpy<float>(N, alpha, X, Y);
}

template<>
void gpu_axpby<double>(const int N, const double alpha, const double *X,
                       const double beta, double *Y) {
  gpu_scale<double>(N, beta, Y);
  gpu_axpy<double>(N, alpha, X, Y);
}

template<>
void gpu_dot<float>(const int n, const float *x, const float *y,
                    float *out) {
  CUBLAS_CHECK(cublasSdot(Config::cublas_handle(), n, x, 1, y, 1, out));
}

template<>
void gpu_dot<double>(const int n, const double *x, const double *y,
                     double *out) {
  CUBLAS_CHECK(cublasDdot(Config::cublas_handle(), n, x, 1, y, 1, out));
}

template<>
void gpu_asum<float>(const int n, const float *x, float *y) {
  CUBLAS_CHECK(cublasSasum(Config::cublas_handle(), n, x, 1, y));
}

template<>
void gpu_asum<double>(const int n, const double *x, double *y) {
  CUBLAS_CHECK(cublasDasum(Config::cublas_handle(), n, x, 1, y));
}

//template<>
//void gpu_scale<float>(const int n, const float alpha, const float *x,
//                      float *y) {
//  CUBLAS_CHECK(cublasScopy(Config::cublas_handle(), n, x, 1, y, 1));
//  CUBLAS_CHECK(cublasSscal(Config::cublas_handle(), n, &alpha, y, 1));
//}
//
//template<>
//void gpu_scale<double>(const int n, const double alpha, const double *x,
//                       double *y) {
//  CUBLAS_CHECK(cublasDcopy(Config::cublas_handle(), n, x, 1, y, 1));
//  CUBLAS_CHECK(cublasDscal(Config::cublas_handle(), n, &alpha, y, 1));
//}

template<typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template<typename Dtype>
void gpu_set(const int N, const Dtype alpha, Dtype *Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(stensor/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel < Dtype ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void gpu_set<int>(const int N, const int alpha, int *Y);
template void gpu_set<float>(const int N, const float alpha, float *Y);
template void gpu_set<double>(const int N, const double alpha, double *Y);

template<typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype *x, const Dtype alpha, Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = x[index] + alpha;
  }
}

template<typename Dtype>
__global__ void add_scalar_kernel_(const int n, const Dtype alpha, Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template<>
void gpu_add_scalar(const int N, const float *X, const float alpha, float *Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  if (X == Y)
    add_scalar_kernel_ < float ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, alpha, Y);
  else add_scalar_kernel < float ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, X, alpha, Y);
}

template<>
void gpu_add_scalar(const int N, const double *X, const double alpha, double *Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  if (X == Y)
    add_scalar_kernel_ < double ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, alpha, Y);
  else add_scalar_kernel < double ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, X, alpha, Y);
}

template<typename Dtype>
__global__ void add_kernel(const int n, const Dtype *a,
                           const Dtype *b, Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template<>
void gpu_add<float>(const int N, const float *a, const float *b,
                    float *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel < float ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template<>
void gpu_add<double>(const int N, const double *a, const double *b,
                     double *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel < double ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template<typename Dtype>
__global__ void sub_kernel(const int n, const Dtype *a,
                           const Dtype *b, Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template<>
void gpu_sub<float>(const int N, const float *a, const float *b,
                    float *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel < float ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template<>
void gpu_sub<double>(const int N, const double *a, const double *b,
                     double *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel < double ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template<typename Dtype>
__global__ void mul_kernel(const int n, const Dtype *a,
                           const Dtype *b, Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template<>
void gpu_mul<float>(const int N, const float *a,
                    const float *b, float *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel < float ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template<>
void gpu_mul<double>(const int N, const double *a,
                     const double *b, double *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel < double ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template<typename Dtype>
__global__ void div_kernel(const int n, const Dtype *a,
                           const Dtype *b, Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template<>
void gpu_div<float>(const int N, const float *a,
                    const float *b, float *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel < float ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template<>
void gpu_div<double>(const int N, const double *a,
                     const double *b, double *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel < double ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template<typename Dtype>
__global__ void abs_kernel(const int n, const Dtype *a, Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template<>
void gpu_abs<float>(const int N, const float *a, float *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel < float ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, y);
}

template<>
void gpu_abs<double>(const int N, const double *a, double *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel < double ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, y);
}

template<typename Dtype>
__global__ void exp_kernel(const int n, const Dtype *a, Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template<>
void gpu_exp<float>(const int N, const float *a, float *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel < float ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, y);
}

template<>
void gpu_exp<double>(const int N, const double *a, double *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel < double ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, y);
}

template<typename Dtype>
__global__ void log_kernel(const int n, const Dtype *a, Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template<>
void gpu_log<float>(const int N, const float *a, float *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel < float ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, y);
}

template<>
void gpu_log<double>(const int N, const double *a, double *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel < double ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, y);
}

template<typename Dtype>
__global__ void pow_scalar_kernel(const int n, const Dtype *a,
                                  const Dtype alpha, Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template<>
void gpu_pow_scalar<float>(const int N, const float *a,
                           const float alpha, float *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  pow_scalar_kernel < float ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template<>
void gpu_pow_scalar<double>(const int N, const double *a,
                            const double alpha, double *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  pow_scalar_kernel < double ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template<typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype *a, Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sqrt(a[index]);
  }
}

template<>
void gpu_sqrt<float>(const int N, const float *a, float *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel < float ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, y);
}

template<>
void gpu_sqrt<double>(const int N, const double *a, double *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel < double ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
    - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void gpu_rng_uniform(const int n, unsigned int *r) {
  CURAND_CHECK(curandGenerate(Config::curand_generator(), r, n));
}

template<>
void gpu_rng_uniform<float>(const int n, const float a, const float b,
                            float *r) {
  CURAND_CHECK(curandGenerateUniform(Config::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    gpu_scale(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    gpu_add_scalar(n, r, a, r);
  }
}

template<>
void gpu_rng_uniform<double>(const int n, const double a, const double b,
                             double *r) {
  CURAND_CHECK(curandGenerateUniformDouble(Config::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    gpu_scale(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    gpu_add_scalar(n, r, a, r);
  }
}

template<>
void gpu_rng_gaussian(const int n, const float mu, const float sigma,
                      float *r) {
  CURAND_CHECK(
      curandGenerateNormal(Config::curand_generator(), r, n, mu, sigma));
}

template<>
void gpu_rng_gaussian(const int n, const double mu, const double sigma,
                      double *r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Config::curand_generator(), r, n, mu, sigma));
}
#define MIN(a, b, c) c = a>b ? b: a
#define MAXAXES 32

#define BROADCAST_INDEX(index, n, sy, shape_a, shape_b, shape_y) \
  int indices_in_result[MAXAXES]{0};\
  indices_in_result[sy - 1] = index % shape_y[sy - 1]; \
  int div = 1; \
  for (int i = sy - 2; i >= 0; --i) { \
    div *=  shape_y[i + 1]; \
    indices_in_result[i] = (index / div)% shape_y[i]; \
  } \
  int index_a = 0; \
  int index_b = 0; \
  for (int i = 0; i < sy; ++i) { \
    int ma;           \
    MIN(indices_in_result[i], shape_a[i], ma); \
    int mb;        \
    MIN(indices_in_result[i], shape_b[i], mb);  \
    index_a *= shape_a[i]; \
    index_b *= shape_b[i]; \
    if (shape_a[i]!=1)  index_a += ma; \
    if (shape_b[i]!=1)  index_b += mb; \
  }

#define IMPLEMENT_BROADCAST_OP_KERNEL(name, op)\
template<typename Dtype>\
__global__ void name##_broadcast_kernel(const int n, const int sy,\
                                     const Dtype *a, const Dtype *b,\
                                     const int *shape_a, const int *shape_b,\
                                     const int *shape_y, Dtype *y) {\
  CUDA_KERNEL_LOOP(index, n) {\
    BROADCAST_INDEX(index, n, sy, shape_a, shape_b, shape_y);\
    y[index] = a[index_a] op b[index_b];\
  }\
}

IMPLEMENT_BROADCAST_OP_KERNEL(add, +);
IMPLEMENT_BROADCAST_OP_KERNEL(sub, -);
IMPLEMENT_BROADCAST_OP_KERNEL(mul, *);
IMPLEMENT_BROADCAST_OP_KERNEL(div, /);

#define IMPLEMENT_BINARY_BROADCAST_GPU_FUNC(name, type) \
template<>\
void gpu_##name##_broadcast<type>(const type *a, const type *b,\
                              std::vector<uint32_t> &shape_a,\
                              std::vector<uint32_t> &shape_b,\
                              type *y) {\
  std::vector<uint32_t> shape_y = stensor::broadcast(shape_a, shape_b);\
  int sy = shape_y.size();\
  int *shape_a_gpu;\
  int *shape_b_gpu;\
  int *shape_y_gpu;\
  MallocGPU((void **) &shape_a_gpu, sy * sizeof(int));\
  MallocGPU((void **) &shape_b_gpu, sy * sizeof(int));\
  MallocGPU((void **) &shape_y_gpu, sy * sizeof(int));\
  int n = 1;\
  for (int i = 0; i < sy; ++i) n *= static_cast<int>(shape_y[i]);\
  for (int i = 0; i < sy; ++i) shape_a_gpu[i] = static_cast<int>(shape_a[i]);\
  for (int i = 0; i < sy; ++i) shape_b_gpu[i] = static_cast<int>(shape_b[i]);\
  for (int i = 0; i < sy; ++i) shape_y_gpu[i] = static_cast<int>(shape_y[i]);\
  name##_broadcast_kernel < type ><<<GET_BLOCKS(n), CUDA_NUM_THREADS>>>(\
      n, sy, a, b, shape_a_gpu, shape_b_gpu, shape_y_gpu, y);\
  FreeGPU(shape_a_gpu);\
  FreeGPU(shape_b_gpu);\
  FreeGPU(shape_y_gpu);\
}

IMPLEMENT_BINARY_BROADCAST_GPU_FUNC(add, float);
IMPLEMENT_BINARY_BROADCAST_GPU_FUNC(add, double);

IMPLEMENT_BINARY_BROADCAST_GPU_FUNC(sub, float);
IMPLEMENT_BINARY_BROADCAST_GPU_FUNC(sub, double);

IMPLEMENT_BINARY_BROADCAST_GPU_FUNC(mul, float);
IMPLEMENT_BINARY_BROADCAST_GPU_FUNC(mul, double);

IMPLEMENT_BINARY_BROADCAST_GPU_FUNC(div, float);
IMPLEMENT_BINARY_BROADCAST_GPU_FUNC(div, double);

}
