/**
* Copyright 2021 wss
* Created by wss on 11月,21, 2021
*/
#ifndef STENSOR_CORE_UTILS_HPP_
#define STENSOR_CORE_UTILS_HPP_
#include "tensor.hpp"
#include <iostream>

namespace stensor {
int get_index(std::vector<int> &shape, Tensor::PairIndexType &start_ends, int index) {
  // start_ends must be ascending
  std::vector<int> new_shape(shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    new_shape[i] = start_ends[i].end - start_ends[i].start;
  }
  int num_axis = new_shape.size();
  std::vector<int> indices(shape.size());
  int div = 1;
  indices[num_axis - 1] = index % new_shape[num_axis - 1];
  for (int i = num_axis - 2; i >= 0; --i) {
    div *= new_shape[i + 1];
    indices[i] = (index / div) % new_shape[i];
  }
  for (int i = 0; i < num_axis; ++i) {
    indices[i] += start_ends[i].start;
  }
  int offset = 0;
  for (int i = 0; i < num_axis; ++i) {
    offset *= shape[i];
    if (indices.size() > i) {
      CHECK_LT(indices[i], shape[i]);
      offset += indices[i];
    }
  }
  return offset;
}
}
#endif //STENSOR_CORE_UTILS_HPP_
