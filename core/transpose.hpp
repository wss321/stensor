/**
* Copyright 2021 wss
* Created by wss on 11月,21, 2021
*/
#ifndef STENSOR_CORE_TRANSPOSE_HPP_
#define STENSOR_CORE_TRANSPOSE_HPP_
#include "tensor.hpp"

namespace stensor {
template<typename Dtype>
Tensor<Dtype> *transpose(Tensor<Dtype> *tensor, std::vector<int> order);
void transpose_gpu(const std::vector<int> shape,
                   const std::vector<int> &stride_x_cpu,
                   const std::vector<int> &stride_y_cpu,
                   const std::vector<int> &order,
                   const float *x,
                   float *y);
}
#endif //STENSOR_CORE_TRANSPOSE_HPP_
