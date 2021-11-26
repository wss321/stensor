/**
* Copyright 2021 wss
* Created by wss on 11月,26, 2021
*/
#include "leaky_relu_layer.hpp"
#include "math/math_base_cuda.hpp"

namespace stensor {

namespace nn {

template<typename Dtype>
__global__ void leakyrelu_backward_kernel(const int n, const Dtype *y_grad, const Dtype *x_data, Dtype *x_grad, Dtype alpha) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype x=x_data[index];
    x_grad[index] += y_grad[index]*(x > 0 ? Dtype(1.0): alpha);
  }
}

void LeakyReLU::backward_gpu() {
  SharedTensor x(inputs_[0]);
  SharedTensor y(outputs_[0]);
  const float *y_grad = y->const_grad();
  const float *x_data = x->const_data();
  float *x_grad = x->grad();

  leakyrelu_backward_kernel<float><<<GET_BLOCKS(x->size()),
  CUDA_NUM_THREADS>>>(x->size(), y_grad, x_data, x_grad, alpha_);

  inputs_.clear();
}

}//namespace nn

}//namespace stensor