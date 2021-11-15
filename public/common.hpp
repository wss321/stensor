#ifndef STENSOR_COMMON_HPP
#define STENSOR_COMMON_HPP
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>

#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

#define DISABLE_COPY_AND_ASSIGN(classname) \
 private:                                  \
  classname(const classname&);             \
  classname& operator=(const classname&)

#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented."

namespace stensor {
enum Mode { CPU, GPU };

template<typename RepeatType, typename V>
inline void RepeatTypeToVector(const RepeatType &proto_data, std::vector<V> &target) {
  target.resize(proto_data.size());
  for (int i = 0; i < proto_data.size(); ++i) {
    target.push_back(static_cast<V>(proto_data[i]));
  }
}
uint32_t count(const std::vector<uint32_t> &shape);

std::vector<uint32_t> broadcast(const std::vector<uint32_t> &shape1, const std::vector<uint32_t> &shape2);

template<typename Dtype>
std::ostream &operator<<(std::ostream &out, std::vector<Dtype> vector);

}// namespace stensor
#endif //STENSOR_COMMON_HPP
