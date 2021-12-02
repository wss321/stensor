/**
* Copyright 2021 wss
* Created by wss on 11月,21, 2021
*/
#ifndef STENSOR_CORE_MATH_TENSOR_BACKWARD_HPP_
#define STENSOR_CORE_MATH_TENSOR_BACKWARD_HPP_
#include "public/common.hpp"
#include "tensor.hpp"

namespace stensor{

namespace backward{
template<typename Dtype>
void matmul_backward(Tensor<Dtype> *a, Tensor<Dtype> *b, const Tensor<Dtype> *y, int axis = -1, bool transA = false, bool transB = false);
// y = a + b
template<typename Dtype>
void add_backward(Tensor<Dtype> *a, Tensor<Dtype> *b, const Tensor<Dtype> *y);

}

}
#endif //STENSOR_CORE_MATH_TENSOR_BACKWARD_HPP_
