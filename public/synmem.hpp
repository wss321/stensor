/**
* Created by wss on 11月,14, 2021
*/
#ifndef STENSOR_INCLUDE_SYNMEM_HPP_
#define STENSOR_INCLUDE_SYNMEM_HPP_
#include <cstdlib>
#include "common.hpp"

namespace stensor {

inline void MallocCPU(void **ptr, uint32_t size) {
  *ptr = std::malloc(size);
  CHECK(*ptr) << "[CPU] Allocate Failed:" << size << "Byte.";
}
inline void FreeCPU(void *ptr) {
  std::free(ptr);
}

class SynMem {
 public:
  SynMem();
  explicit SynMem(uint32_t size);
  ~SynMem();
  const void *cpu_data();
//  const void * gpu_data();
  void set_cpu_data(void *data_ptr);
//  void set_gpu_data(void* data);
  void *mutable_cpu_data();
//  void* mutable_gpu_data();

  enum SynState { NONE, AT_CPU, AT_GPU, SYNCED };
  uint32_t size() const { return size_; }
  SynState state() const { return state_; }

 private:
  void to_cpu();
  void syn();
  void *cpu_ptr_;
  uint32_t size_;
  SynState state_;
  bool own_cpu_data_;

 DISABLE_COPY_AND_ASSIGN(SynMem);
};

}
#endif //STENSOR_INCLUDE_SYNMEM_HPP_
