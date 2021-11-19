/**
* Copyright 2021 wss
* Created by wss on 11月,16, 2021
*/

#include "math_base_cuda.hpp"

namespace stensor {
#define MIN_FUNC(a, b, c) c = a>b ? b: a

/* self-op start*/

#define IMPLEMENT_GPU_UNARY_FUNC(name, op_expression) \
template<typename Dtype> \
__global__ void name##_kernel(const int n, const Dtype* x, Dtype* y) { \
  CUDA_KERNEL_LOOP(index, n) { \
    op_expression; \
  } \
} \
template <> \
void gpu_##name<float>(const int n, const float* x, float* y) { \
  name##_kernel<float><<<GET_BLOCKS(n), CUDA_NUM_THREADS>>>( \
      n, x, y); \
} \
template <> \
void gpu_##name<double>(const int n, const double* x, double* y) { \
  name##_kernel<double><<<GET_BLOCKS(n), CUDA_NUM_THREADS>>>( \
      n, x, y); \
}

IMPLEMENT_GPU_UNARY_FUNC(exp, y[index] = exp(x[index]));
IMPLEMENT_GPU_UNARY_FUNC(log, y[index] = log(x[index]));
IMPLEMENT_GPU_UNARY_FUNC(abs, y[index] = abs(x[index]));
IMPLEMENT_GPU_UNARY_FUNC(sqrt, y[index] = sqrt(x[index]));
IMPLEMENT_GPU_UNARY_FUNC(square, y[index] = x[index] * x[index]);
IMPLEMENT_GPU_UNARY_FUNC(sign, y[index] = x[index] > Dtype(0.0) ? Dtype(1.0) : -Dtype(1.0));
IMPLEMENT_GPU_UNARY_FUNC(sigmoid, y[index] = Dtype(1.0) / (Dtype(1.0) + exp(-x[index])))
//IMPLEMENT_GPU_UNARY_FUNC(tanh, Dtype e_2x = exp(Dtype(2.0) * x[index]);y[index] = (e_2x-Dtype(1.0))/(e_2x+Dtype(1.0)));
IMPLEMENT_GPU_UNARY_FUNC(tanh, y[index] = tanh(x[index]));
IMPLEMENT_GPU_UNARY_FUNC(relu, y[index] = x[index] > Dtype(0.0) ? x[index] : Dtype(0.0));
IMPLEMENT_GPU_UNARY_FUNC(elu, y[index] = x[index] > Dtype(0.0) ? x[index] : exp(x[index]) - Dtype(1.0));
IMPLEMENT_GPU_UNARY_FUNC(gelu, y[index] = Dtype(0.5) * x[index] * (Dtype(1.0) + erf(x[index] / sqrt(Dtype(2.0)))));
IMPLEMENT_GPU_UNARY_FUNC(leakyrelu, y[index] = x[index] > 0 ? x[index] : Dtype(0.2) * x[index]);
IMPLEMENT_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

template<>
void gpu_asum<float>(const int n, const float *x, float *y) {
  CUBLAS_CHECK(cublasSasum(Config::cublas_handle(), n, x, 1, y));
}

template<>
void gpu_asum<double>(const int n, const double *x, double *y) {
  CUBLAS_CHECK(cublasDasum(Config::cublas_handle(), n, x, 1, y));
}

template<typename Dtype>
__global__ void clamp_kernel(const int n,
                             const Dtype min, const Dtype max,
                             const Dtype *x, Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) {
    if (x[index] >= min && x[index] <= max) y[index] = x[index];
    else if (x[index] < min) y[index] = min;
    else if (x[index] > max) y[index] = max;
  }
}
template<typename Dtype>
void gpu_clamp(const int n,
               const Dtype min, const Dtype max,
               const Dtype *x,
               Dtype *y) {
  clamp_kernel<Dtype><<<GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      n, min, max, x, y);
}
template void gpu_clamp<int>(const int n, const int min, const int max, const int *x, int *y);
template void gpu_clamp<float>(const int n, const float min, const float max, const float *x, float *y);
template void gpu_clamp<double>(const int n, const double min, const double max, const double *x, double *y);

/* self-op end*/

/* vector-scalar start*/

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


/* vector-scalar end*/

/* vector-vector start*/

#define IMPLEMENT_GPU_BINARY_FUNC(name, op_expression) \
template<typename Dtype>\
__global__ void name##_kernel(const int n, const Dtype *a,\
                           const Dtype *b, Dtype *y) {\
  CUDA_KERNEL_LOOP(index, n) {\
    op_expression;\
  }\
}\
template<>\
void gpu_##name<int>(const int N, const int *a, const int *b,\
                    int *y) { \
  name##_kernel<int><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(\
      N, a, b, y);\
}\
template<>\
void gpu_##name<float>(const int N, const float *a, const float *b,\
                    float *y) { \
  name##_kernel < float ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(\
      N, a, b, y);\
}\
template<>\
void gpu_##name<double>(const int N, const double *a, const double *b,\
                     double *y) {\
  name##_kernel < double ><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(\
      N, a, b, y);\
}

void gpu_copy(const size_t N, const void *X, void *Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(stensor/alt_fn)
  }
}

IMPLEMENT_GPU_BINARY_FUNC(add, y[index] = a[index] + b[index]);
IMPLEMENT_GPU_BINARY_FUNC(sub, y[index] = a[index] - b[index]);
IMPLEMENT_GPU_BINARY_FUNC(mul, y[index] = a[index] * b[index]);
IMPLEMENT_GPU_BINARY_FUNC(div, y[index] = a[index] / b[index]);
IMPLEMENT_GPU_BINARY_FUNC(pow, y[index] = pow(a[index], b[index]));

// Broadcast functions

#define BROADCAST_INDEX(index, n, sy, shape_a, shape_b, shape_y) \
  int indices_in_result[MAX_AXES]{0};\
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
    MIN_FUNC(indices_in_result[i], shape_a[i], ma); \
    int mb;        \
    MIN_FUNC(indices_in_result[i], shape_b[i], mb);  \
    index_a *= shape_a[i]; \
    index_b *= shape_b[i]; \
    if (shape_a[i]!=1)  index_a += ma; \
    if (shape_b[i]!=1)  index_b += mb; \
  }

#define IMPLEMENT_BROADCAST_OP_KERNEL(name, op_expression)\
template<typename Dtype>\
__global__ void name##_kernel(const int n, const int sy,\
                                     const Dtype *a, const Dtype *b,\
                                     const int *shape_a, const int *shape_b,\
                                     const int *shape_y, Dtype *y) {\
  CUDA_KERNEL_LOOP(index, n) {\
    BROADCAST_INDEX(index, n, sy, shape_a, shape_b, shape_y);\
    op_expression;\
  }\
}

#define IMPLEMENT_BINARY_BROADCAST_GPU_FUNC(name, type) \
template<>\
void gpu_##name<type>(const type *a, const type *b,\
                              std::vector<int> &shape_a,\
                              std::vector<int> &shape_b,\
                              type *y) {\
  std::vector<int> shape_y = stensor::broadcast(shape_a, shape_b);\
  int sy = shape_y.size();\
  int *shape_a_gpu;\
  int *shape_b_gpu;\
  int *shape_y_gpu;\
  MallocGPU((void **) &shape_a_gpu, sy * sizeof(int));\
  MallocGPU((void **) &shape_b_gpu, sy * sizeof(int));\
  MallocGPU((void **) &shape_y_gpu, sy * sizeof(int));\
  int n = 1;\
  for (int i = 0; i < sy; ++i) n *= shape_y[i];\
  for (int i = 0; i < sy; ++i) shape_a_gpu[i] = shape_a[i];\
  for (int i = 0; i < sy; ++i) shape_b_gpu[i] = shape_b[i];\
  for (int i = 0; i < sy; ++i) shape_y_gpu[i] = shape_y[i];\
  name##_kernel < type ><<<GET_BLOCKS(n), CUDA_NUM_THREADS>>>(\
      n, sy, a, b, shape_a_gpu, shape_b_gpu, shape_y_gpu, y);\
  FreeGPU(shape_a_gpu);\
  FreeGPU(shape_b_gpu);\
  FreeGPU(shape_y_gpu);\
}

IMPLEMENT_BROADCAST_OP_KERNEL(add_broadcast, y[index] = a[index_a] + b[index_b]);
IMPLEMENT_BROADCAST_OP_KERNEL(sub_broadcast, y[index] = a[index_a] - b[index_b]);
IMPLEMENT_BROADCAST_OP_KERNEL(mul_broadcast, y[index] = a[index_a] * b[index_b]);
IMPLEMENT_BROADCAST_OP_KERNEL(div_broadcast, y[index] = a[index_a] / b[index_b]);
IMPLEMENT_BROADCAST_OP_KERNEL(pow_broadcast, y[index] = pow(a[index_a], b[index_b]));

IMPLEMENT_BINARY_BROADCAST_GPU_FUNC(add_broadcast, float);
IMPLEMENT_BINARY_BROADCAST_GPU_FUNC(add_broadcast, double);

IMPLEMENT_BINARY_BROADCAST_GPU_FUNC(sub_broadcast, float);
IMPLEMENT_BINARY_BROADCAST_GPU_FUNC(sub_broadcast, double);

IMPLEMENT_BINARY_BROADCAST_GPU_FUNC(mul_broadcast, float);
IMPLEMENT_BINARY_BROADCAST_GPU_FUNC(mul_broadcast, double);

IMPLEMENT_BINARY_BROADCAST_GPU_FUNC(div_broadcast, float);
IMPLEMENT_BINARY_BROADCAST_GPU_FUNC(div_broadcast, double);

IMPLEMENT_BINARY_BROADCAST_GPU_FUNC(pow_broadcast, float);
IMPLEMENT_BINARY_BROADCAST_GPU_FUNC(pow_broadcast, double);

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
void gpu_stride_dot<float>(int n,
                           const float *x, int incx,
                           const float *y, int incy,
                           float *out) {
  CUBLAS_CHECK(cublasSdot(Config::cublas_handle(), n, x, incx, y, incy, out));
}
template<>
void gpu_stride_dot<double>(int n,
                            const double *x, int incx,
                            const double *y, int incy,
                            double *out) {
  CUBLAS_CHECK(cublasDdot(Config::cublas_handle(), n, x, incx, y, incy, out));
}


/* vector-vector end*/

/* matrix-vector start*/

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

/* matrix-vector end*/

/* matrix-matrix start*/
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
void gpu_axpy<float>(const int N, const float alpha, const float *X,
                     float *Y) {
  CUBLAS_CHECK(cublasSaxpy(Config::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template<>
void gpu_axpy<double>(const int N, const double alpha, const double *X,
                      double *Y) {
  CUBLAS_CHECK(cublasDaxpy(Config::cublas_handle(), N, &alpha, X, 1, Y, 1));
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

/* matrix-matrix end*/

/* random generator start*/

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
/* random generator end*/


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

}
