/**
* Created by wss on 11月,14, 2021
*/
#include "synmem.hpp"
#include "common.hpp"

namespace stensor {

SynMem::SynMem() :
    cpu_ptr_(nullptr),
    size_(0ul),
    state_(NONE),
    own_cpu_data_(false) {}

SynMem::SynMem(uint32_t size) :
    cpu_ptr_(nullptr),
    size_(size),
    state_(NONE),
    own_cpu_data_(false) {}

SynMem::~SynMem() {
  if (cpu_ptr_ && own_cpu_data_)
    FreeCPU(cpu_ptr_);
}

void SynMem::to_cpu() {
  switch (state_) {
    case NONE:MallocCPU(&cpu_ptr_, size_);
      std::memset(cpu_ptr_, 0, size_);
      state_ = AT_CPU;
      own_cpu_data_ = true;
      break;
    case AT_CPU:
    case AT_GPU:
    case SYNCED:break;
  }
}

void SynMem::syn() {
  //TODO:syn
}

const void* SynMem::cpu_data() {
  to_cpu();
  return const_cast<const void *> (cpu_ptr_);
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

void *SynMem::mutable_cpu_data() {
  to_cpu();
  state_ = AT_CPU;
  return cpu_ptr_;
}

}