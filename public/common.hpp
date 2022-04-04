#ifndef STENSOR_COMMON_HPP
#define STENSOR_COMMON_HPP
#include <gflags/gflags.h>
#include <glog/logging.h>

//#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>
#include<sys/timeb.h>

// cuda


#include "stensor_random.hpp"
#include "memory_op.hpp"
#include <boost/thread.hpp>
//#include "cudnn.h"

#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

namespace stensor {
#define MAX_AXES 32

#define SCINT static_cast<int>

#define DISABLE_COPY_AND_ASSIGN(classname) \
 private:                                  \
  classname(const classname&);             \
  classname& operator=(const classname&)

#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented"

enum Mode { CPU, GPU };

template<typename RepeatType, typename V>
inline void RepeatTypeToVector(const RepeatType &proto_data, std::vector<V> &target) {
  target.resize(proto_data.size());
  for (int i = 0; i < proto_data.size(); ++i) {
    target[i] = static_cast<V>(proto_data[i]);
  }
}
uint32_t count(const std::vector<uint32_t> &shape);

template<typename Dshape>
std::vector<Dshape> broadcast(std::vector<Dshape> &shape1, std::vector<Dshape> &shape2);

template<typename Dtype>
std::ostream &operator<<(std::ostream &out, std::vector<Dtype> vector);

/* borrow from caffe */

class Config {
 public:
  ~Config();
  static Config &GetInstance();
  inline static bool multiprocess() { return GetInstance().multiprocess_; }
  inline static cublasHandle_t cublas_handle() { return GetInstance().cublas_handle_; }
//  inline static cudnnHandle_t cudnn_handle(){return  GetInstance().cudnn_handle_;}
  inline static curandGenerator_t curand_generator() {
    return GetInstance().curand_generator_;
  }
  inline static void set_multiprocess(bool val) { GetInstance().multiprocess_ = val; }
  static void set_random_seed(const unsigned int seed);
  inline static RNG &rng_stream() {
    if (!GetInstance().random_generator_) {
      GetInstance().random_generator_.reset(new RNG());
    }
    return *(GetInstance().random_generator_);
  }

 protected:
//  cudnnHandle_t cudnn_handle_;
  cublasHandle_t cublas_handle_;
  curandGenerator_t curand_generator_;
  std::shared_ptr<RNG> random_generator_;
  bool multiprocess_;

 private:
  // The private constructor to avoid duplicate instantiation.
  Config();

 DISABLE_COPY_AND_ASSIGN(Config);
};

inline rng_t *global_rng() {
  return static_cast<stensor::rng_t *>(Config::rng_stream().generator());
}

inline long long systemtime_ms() {
  timeb t;
  ftime(&t);
  return t.time * 1000 + t.millitm;
}

inline void swap(int &a, int &b) {
  a = a ^ b;
  b = a ^ b;
  a = a ^ b;
}

/* borrow from caffe /end*/
}// namespace stensor
#endif //STENSOR_COMMON_HPP
