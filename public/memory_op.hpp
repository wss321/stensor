#ifndef STENSOR_DEVICES_H
#define STENSOR_DEVICES_H
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <driver_types.h>  // cuda driver types
#include <cublas_v2.h>
#include <glog/logging.h>
/*
 * CUDA macros
 * */
// CUDA: various checks for different function calls.

namespace stensor {

const char* cublasGetErrorString(cublasStatus_t error);

const char *curandGetErrorString(curandStatus_t error);

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << stensor::curandGetErrorString(status); \
  } while (0)


// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

inline int getMaxThreadNum() {
  cudaDeviceProp prop;
//  int num_gpu;
//  CUDA_CHECK(cudaGetDeviceCount(&num_gpu));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  return prop.maxThreadsPerBlock;
}

//#define CUDA_NUM_THREADS getMaxThreadNum()
#define CUDA_NUM_THREADS 512

inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << stensor::cublasGetErrorString(status); \
  } while (0)
inline void MallocCPU(void **ptr, uint32_t size) {
  *ptr = std::malloc(size);
  CHECK(*ptr) << "[CPU] Allocate Failed:" << size << "Byte.";
}

inline void FreeCPU(void *ptr) {
  std::free(ptr);
}

//#ifndef CPU_ONLY
inline void MallocGPU(void **ptr, uint32_t size) {
  CUDA_CHECK(cudaMallocHost(ptr, size));
}

inline void FreeGPU(void *ptr) {
  CUDA_CHECK(cudaFreeHost(ptr));
}

inline void memcopy(const size_t N, const void *X, void *Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));
  }
}
inline void gpu_memset(const size_t N, const int alpha, void *X) {
  CUDA_CHECK(cudaMemset(X, alpha, N));
}

inline void cpu_memset(const size_t N, const int alpha, void *X) {
  std::memset(X, alpha, N);  // NOLINT(caffe/alt_fn)
}

//#endif
}//namespace stensor

#endif //STENSOR_DEVICES_H
