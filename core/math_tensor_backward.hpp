/**
* Copyright 2021 wss
* Created by wss on 11月,21, 2021
*/
#ifndef STENSOR_CORE_MATH_TENSOR_BACKWARD_HPP_
#define STENSOR_CORE_MATH_TENSOR_BACKWARD_HPP_
#include "public/common.hpp"
#include "tensor.hpp"
#include "math_base_cpu.hpp"
#include "math_base_cuda.hpp"
#include "utils.hpp"
namespace stensor{

namespace backward{

void matmul_backward(Tensor *a, Tensor *b, const Tensor *y, int axis = -1, bool transA = false, bool transB = false);
// y = a + b
void add_backward(Tensor *a, Tensor *b, const Tensor *y);

}

}
#endif //STENSOR_CORE_MATH_TENSOR_BACKWARD_HPP_
