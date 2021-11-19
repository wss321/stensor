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
  SynMem();
//  explicit SynMem(uint32_t size);
  explicit SynMem(uint32_t size, int device_id = -1);
  ~SynMem();
  enum SynState { NONE, AT_CPU, AT_GPU, BOTH };
  inline uint32_t size() const { return size_; }
  inline SynState state() const { return state_; }
  inline int device() const { return gpu_device_; }
  inline bool has_gpu_data() const { return gpu_ptr_ && own_gpu_data_; }
  inline bool has_cpu_data() const { return cpu_ptr_ && own_cpu_data_; }
  inline const void *cpu_data() {
    CHECK(has_cpu_data()) << "CPU data is none";
    return const_cast<const void *> (cpu_ptr_);
  }

  const void *gpu_data() {
    CHECK(has_gpu_data()) << "GPU data is none";
    return const_cast<const void *> (gpu_ptr_);
  }

  void set_cpu_data(void *data_ptr);
  void set_gpu_data(void *data_ptr);
  inline void *mutable_cpu_data() {
    CHECK(has_cpu_data()) << "CPU data is none";
    return cpu_ptr_;
  };
  inline void *mutable_gpu_data() {
    CHECK(has_gpu_data()) << "GPU data is none";
    return gpu_ptr_;
  };

//  void to_cpu();
//  void to_gpu();
//  void syn();
  inline void copy_cpu_to_gpu() {
    CHECK(has_cpu_data()) << "CPU data is none";
    CHECK(has_gpu_data()) << "GPU data is none";
    stensor::memcopy(size_, cpu_ptr_, gpu_ptr_);
  };
  inline void copy_gpu_to_cpu() {
    CHECK(has_cpu_data()) << "CPU data is none";
    CHECK(has_gpu_data()) << "GPU data is none";
    stensor::memcopy(size_, gpu_ptr_, cpu_ptr_);
  };
  inline void free_cpu() {
    if (has_cpu_data())
      FreeCPU(cpu_ptr_);
    cpu_ptr_ = nullptr;
    own_cpu_data_ = false;
    update_state();
  };
  inline void free_gpu() {
    if (has_gpu_data())
      FreeGPU(gpu_ptr_);
    gpu_ptr_ = nullptr;
    own_gpu_data_ = false;
    update_state();
  };
  inline void alloc_cpu() {
    if (has_cpu_data()) {
      LOG(INFO) << "Already alloc cpu";
      return;
    }

    MallocCPU(&cpu_ptr_, size_);
    own_cpu_data_ = true;
    update_state();
  };
  inline void alloc_gpu() {
    if (has_gpu_data()) {
      LOG(INFO) << "Already alloc gpu";
      return;
    }
    MALLOC_GPU_DEVICE(gpu_ptr_, size_, gpu_device_);
    own_gpu_data_ = true;
    update_state();
  };
 private:
  void *cpu_ptr_;
  void *gpu_ptr_;
  uint32_t size_;
  SynState state_;
  bool own_cpu_data_;
  bool own_gpu_data_;
  int gpu_device_;

  void update_state();

 DISABLE_COPY_AND_ASSIGN(SynMem);
};

}
#endif //STENSOR_INCLUDE_SYNMEM_HPP_
