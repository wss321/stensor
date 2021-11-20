/**
* Created by wss on 11月,14, 2021
*/
#ifndef STENSOR_INCLUDE_SYNMEM_HPP_
#define STENSOR_INCLUDE_SYNMEM_HPP_
#include <cstdlib>
#include "common.hpp"
#include "memory_op.hpp"
#include <ctime>

namespace stensor {

#define MALLOC_GPU_DEVICE(gpu_ptr, size, gpu_device) \
  MallocGPU(&gpu_ptr, size);\
  int global_device;\
  CUDA_CHECK(cudaGetDevice(&global_device));\
  if (gpu_device==-1){ \
    CUDA_CHECK(cudaSetDevice(global_device));\
    gpu_device = global_device;\
  }else{ \
  CUDA_CHECK(cudaSetDevice(gpu_device)); \
  } \
  CUDA_CHECK(cudaGetDevice(&gpu_device)); \
  CUDA_CHECK(cudaSetDevice(global_device))

class SynMem {
 public:
  SynMem() :
      cpu_ptr_(nullptr), gpu_ptr_(nullptr),
      size_(0), gpu_device_(-1) {}
  explicit SynMem(int size, int device_id = -1) :
      cpu_ptr_(nullptr), gpu_ptr_(nullptr),
      size_(size), gpu_device_(device_id) {
    if (device_id > -1) alloc_gpu();
    else alloc_cpu();
  }
  ~SynMem() {
    free_cpu();
    free_gpu();
  }
  inline int size() const { return size_; }
  inline int device() const { return gpu_device_; }
  inline void *cpu_data() {
    return cpu_ptr_;
  }

  inline void *gpu_data() {
    return gpu_ptr_;
  }

  inline void set_cpu_data(void *data_ptr, int size) {
    free_cpu();
    free_gpu();
    cpu_ptr_ = data_ptr;
    size_ = size;
  }

  inline void set_gpu_data(void *data_ptr, int size) {
    free_cpu();
    free_gpu();
    gpu_ptr_ = data_ptr;
    size_ = size;
  }

  inline void copy_cpu_to_gpu() {
    stensor::memcopy(size_, cpu_ptr_, gpu_ptr_);
  };

  inline void copy_gpu_to_cpu() {
    stensor::memcopy(size_, gpu_ptr_, cpu_ptr_);
  };

  inline void free_cpu() {
    if (cpu_ptr_ != nullptr) FreeCPU(cpu_ptr_);
    cpu_ptr_ = nullptr;
  };
  inline void free_gpu() {
    if (gpu_ptr_ != nullptr) FreeGPU(gpu_ptr_);
    gpu_ptr_ = nullptr;
  };

  inline void alloc_cpu(int NewSize = 0) {
    if (cpu_ptr_ != nullptr) {
      LOG(INFO) << "Already alloc cpu";
      return;
    }

    if (NewSize == 0)
      MallocCPU(&cpu_ptr_, size_);
    else {
      MallocCPU(&cpu_ptr_, NewSize);
      size_ = NewSize;
    }
  };
  inline void alloc_gpu(int NewSize = 0) {
    if (gpu_ptr_ != nullptr) {
      LOG(INFO) << "Already alloc gpu";
      return;
    }
    if (NewSize == 0) {
      MALLOC_GPU_DEVICE(gpu_ptr_, size_, gpu_device_);
    } else {
      MALLOC_GPU_DEVICE(gpu_ptr_, NewSize, gpu_device_);
      size_ = NewSize;
    }
  };
 private:
  void *cpu_ptr_;
  void *gpu_ptr_;
  int size_;
  int gpu_device_;

 DISABLE_COPY_AND_ASSIGN(SynMem);
};

}
#endif //STENSOR_INCLUDE_SYNMEM_HPP_
