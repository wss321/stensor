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

SynMem::SynMem(uint32_t size, int device_id) :
    cpu_ptr_(nullptr), gpu_ptr_(nullptr),
    size_(size), state_(NONE),
    own_cpu_data_(false), own_gpu_data_(false),
    gpu_device_(device_id) { if (device_id > -1) alloc_gpu(); else alloc_cpu(); }

SynMem::~SynMem() {
  free_cpu();
  free_gpu();
}

void SynMem::update_state() {
  if (has_cpu_data() && has_gpu_data()) {
    state_ = BOTH;
    return;
  }
  if (has_cpu_data() && !has_gpu_data()) {
    state_ = AT_CPU;
    return;
  }
  if (!has_cpu_data() && has_gpu_data()) {
    state_ = AT_GPU;
    return;
  }
  if (!has_cpu_data() && !has_gpu_data()) {
    state_ = NONE;
    return;
  }
}

//void SynMem::to_cpu() {
//  switch (state_) {
//    case NONE:alloc_cpu();
//      std::memset(cpu_ptr_, 0, size_);
//      break;
//    case AT_GPU:
//      if (!has_cpu_data()) {
//        alloc_cpu();
//      }
//      stensor::memcopy(size_, gpu_ptr_, cpu_ptr_);
//      break;
//    case AT_CPU:;
//    case BOTH:;
//      break;
//  }
//  update_state();
//}
//
//void SynMem::to_gpu() {
//  switch (state_) {
//    case NONE:alloc_gpu();
//      CUDA_CHECK(cudaMemset(gpu_ptr_, 0, size_));
//      break;
//    case AT_CPU:
//      if (!has_gpu_data()) {
//        alloc_gpu();
//      }
//      stensor::memcopy(size_, cpu_ptr_, gpu_ptr_);
//      break;
//    case AT_GPU:;
//    case BOTH:;
//      break;
//  }
//  update_state();
//}
//
//void SynMem::syn() {
//  switch (state_) {
//    case NONE:
//      if (gpu_device_ > -1) {
//        alloc_gpu();
//        CUDA_CHECK(cudaMemset(gpu_ptr_, 0, size_));
//      } else {
//        alloc_cpu();
//        std::memset(cpu_ptr_, 0, size_);
//      }
//      break;
//    case AT_CPU:
//      if (!has_gpu_data()) {
//        alloc_gpu();
//      }
//      stensor::memcopy(size_, cpu_ptr_, gpu_ptr_);
//      break;
//    case AT_GPU:
//      if (!has_cpu_data()) {
//        alloc_cpu();
//      }
//      stensor::memcopy(size_, gpu_ptr_, cpu_ptr_);
//      break;
//    case BOTH:break;
//  }
//  update_state();
//}

void SynMem::set_cpu_data(void *data_ptr) {
  CHECK(data_ptr);
  free_cpu();
  free_gpu();
  cpu_ptr_ = data_ptr;
  own_cpu_data_ = true;
  update_state();
}
void SynMem::set_gpu_data(void *data_ptr) {
  CHECK(data_ptr);
  free_cpu();
  free_gpu();
  gpu_ptr_ = data_ptr;
  own_gpu_data_ = true;
  update_state();
}

}