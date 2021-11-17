/**
* Created by wss on 11月,14, 2021
*/
#include "synmem.hpp"
#include "common.hpp"

namespace stensor {

SynMem::SynMem() :
    cpu_ptr_(nullptr), gpu_ptr_(nullptr),
    size_(0ul), state_(NONE),
    own_cpu_data_(false), own_gpu_data_(false),
    gpu_device_(-1) {}

SynMem::SynMem(uint32_t size) :
    cpu_ptr_(nullptr), gpu_ptr_(nullptr),
    size_(size), state_(NONE),
    own_cpu_data_(false), own_gpu_data_(false),
    gpu_device_(-1) {}

SynMem::~SynMem() {
  if (cpu_ptr_ && own_cpu_data_)
    FreeCPU(cpu_ptr_);
  if (gpu_ptr_ && own_gpu_data_)
    FreeGPU(gpu_ptr_);
}

void SynMem::to_cpu() {
  switch (state_) {
    case NONE:MallocCPU(&cpu_ptr_, size_);
      std::memset(cpu_ptr_, 0, size_);
      state_ = AT_CPU;
      own_cpu_data_ = true;
      break;
    case AT_GPU:
      if (cpu_ptr_ == nullptr) {
        MallocCPU(&cpu_ptr_, size_);
        own_cpu_data_ = true;
      }
      stensor::memcopy(size_, gpu_ptr_, cpu_ptr_);
      state_ = SYNED;
      break;
    case AT_CPU:
    case SYNED:break;
  }
}

void SynMem::to_gpu() {
  switch (state_) {
    case NONE:MallocGPU(&gpu_ptr_, size_);
      CUDA_CHECK(cudaMemset(gpu_ptr_, 0, size_));
      CUDA_CHECK(cudaGetDevice(&gpu_device_));
      own_gpu_data_ = true;
      state_ = AT_GPU;
      break;
    case AT_CPU:
      if (gpu_ptr_ == nullptr) {
        MallocGPU(&gpu_ptr_, size_);
        CUDA_CHECK(cudaGetDevice(&gpu_device_));
        own_gpu_data_ = true;
      }
      stensor::memcopy(size_, cpu_ptr_, gpu_ptr_);
      state_ = SYNED;
      break;
    case AT_GPU:
    case SYNED:break;
  }
}

void SynMem::syn() {
  switch (state_) {
    case NONE:MallocGPU(&gpu_ptr_, size_);
      CUDA_CHECK(cudaMemset(gpu_ptr_, 0, size_));
      CUDA_CHECK(cudaGetDevice(&gpu_device_));
      own_gpu_data_ = true;
      if (cpu_ptr_ == nullptr) {
        MallocCPU(&cpu_ptr_, size_);
        own_cpu_data_ = true;
      }
      stensor::memcopy(size_, gpu_ptr_, cpu_ptr_);
      state_ = SYNED;
      break;
    case AT_CPU:
      if (gpu_ptr_ == nullptr) {
        MallocGPU(&gpu_ptr_, size_);
        CUDA_CHECK(cudaGetDevice(&gpu_device_));
        own_gpu_data_ = true;
      }
      stensor::memcopy(size_, cpu_ptr_, gpu_ptr_);
      state_ = SYNED;
      break;
    case AT_GPU:
      if (cpu_ptr_ == nullptr) {
        MallocCPU(&cpu_ptr_, size_);
        own_cpu_data_ = true;
      }
      stensor::memcopy(size_, gpu_ptr_, cpu_ptr_);
      state_ = SYNED;
      break;
    case SYNED:break;
  }
}

const void *SynMem::cpu_data() {
  to_cpu();
  return const_cast<const void *> (cpu_ptr_);
}
const void *SynMem::gpu_data() {
  to_gpu();
  return const_cast<const void *> (gpu_ptr_);
}

void SynMem::set_cpu_data(void *data_ptr) {
  CHECK(data_ptr);
  if (own_cpu_data_) {
    FreeCPU(cpu_ptr_);
  }
  cpu_ptr_ = data_ptr;
  state_ = AT_CPU;
  own_cpu_data_ = false;
}
void SynMem::set_gpu_data(void *data_ptr) {
  CHECK(data_ptr);
  if (own_gpu_data_) {
    FreeCPU(gpu_ptr_);
  }
  gpu_ptr_ = data_ptr;
  state_ = AT_GPU;
  own_gpu_data_ = false;
}

void *SynMem::mutable_cpu_data() {
  to_cpu();
  state_ = AT_CPU;
  return cpu_ptr_;
}

void *SynMem::mutable_gpu_data() {
  to_gpu();
  state_ = AT_GPU;
  return gpu_ptr_;
}

}