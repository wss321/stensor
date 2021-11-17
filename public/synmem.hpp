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

class SynMem {
 public:
  SynMem();
//  explicit SynMem(uint32_t size);
  explicit SynMem(uint32_t size, int device_id=-1);
  ~SynMem();
  const void *cpu_data();
  const void *gpu_data();
  void set_cpu_data(void *data_ptr);
  void set_gpu_data(void *data_ptr);
  void *mutable_cpu_data();
  void *mutable_gpu_data();

  enum SynState { NONE, AT_CPU, AT_GPU, SYNED };
  inline uint32_t size() const { return size_; }
  inline SynState state() const { return state_; }
  inline int device() const { return gpu_device_; }
  inline bool has_gpu_data() const{return own_gpu_data_;}
  inline bool has_cpu_data() const{return own_cpu_data_;}
  void to_cpu();
  void to_gpu();
  void syn();

 private:
  void *cpu_ptr_;
  void *gpu_ptr_;
  uint32_t size_;
  SynState state_;
  bool own_cpu_data_;
  bool own_gpu_data_;
  int gpu_device_;

 DISABLE_COPY_AND_ASSIGN(SynMem);
};

}
#endif //STENSOR_INCLUDE_SYNMEM_HPP_
