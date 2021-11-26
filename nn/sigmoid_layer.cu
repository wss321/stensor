/**
* Copyright 2021 wss
* Created by wss on 11月,27, 2021
*/
#include "sigmoid_layer.hpp"
#include "math/math_base_cuda.hpp"

namespace stensor {

namespace nn {

template<typename Dtype>
__global__ void sigmoid_backward_kernel(const int n, const Dtype *y_grad, const Dtype *y_data, Dtype *x_grad) {
  CUDA_KERNEL_LOOP(index, n) {
    int y = y_data[index];
    x_grad[index] = (1 - y) * y * y_grad[index];
  }
}

void Sigmoid::backward_gpu() {
  SharedTensor x(inputs_[0]);
  SharedTensor y(outputs_[0]);
  const float *y_grad = y->const_grad();
  const float *y_data = y->const_data();
  float *x_grad = x->grad();

  sigmoid_backward_kernel<float><<<GET_BLOCKS(x->size()),
  CUDA_NUM_THREADS>>>(x->size(), y_grad, y_data, x_grad);

  inputs_.clear();
}

}//namespace nn

}//namespace stensor