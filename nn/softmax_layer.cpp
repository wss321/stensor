/**
* Copyright 2021 wss
* Created by wss on 11月,24, 2021
*/
#include "softmax_layer.hpp"
#include "core/math_tensor_backward.hpp"
#include "math/math_base_cuda.hpp"
#include "core/math_tesnsor.hpp"

namespace stensor {

namespace nn {

SoftmaxLayer::SoftmaxLayer(const std::string &name, int axis, int device) : axis_(axis) {
  if (device > -1) state_ = GPU;
  else state_ = CPU;
  type_ = "Softmax";
  name_ = name;
  parameters_.resize(0);

}

TensorVec SoftmaxLayer::forward(TensorVec &inputs) {
  outputs_.resize(inputs.size());
  inputs_ = inputs;
  for (int i = 0; i < inputs.size(); ++i) {
    Tensor *m = stensor::softmax(inputs[i].get(), axis_);
    outputs_[i].reset(m);
  }
  return outputs_;
}

template<typename Dtype>
void cpu_softmax_backward(const int M,
                          const int D,
                          const int N,
                          const Dtype *y_data,
                          const Dtype *y_grad,
                          Dtype beta,
                          Dtype *x_grad) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      // calc denominator
      Dtype sum_delta_y_mul_y = 0;
      for (int d = 0; d < D; ++d) {
        int index = d * N + n;
        sum_delta_y_mul_y += y_data[index] * y_grad[index];
      }

      if (beta == 0) {
        for (int d = 0; d < D; ++d) {
          int index = d * N + n;
          x_grad[index] = y_data[index]*(Dtype(1.0) - sum_delta_y_mul_y);
        }
      } else {
        for (int d = 0; d < D; ++d) {
          int index = d * N + n;
          x_grad[index] = beta * x_grad[index] + y_data[index]*(Dtype(1.0) - sum_delta_y_mul_y);
        }
      }
    }
    x_grad += D * N;
    y_data += D * N;
    y_grad += D * N;
  }
}

void SoftmaxLayer::backward_cpu() {
  for (int i = 0; i < inputs_.size(); ++i) {
    SharedTensor in = inputs_[i];
    SharedTensor out = outputs_[i];
    if (!in->require_grad()||!out->require_grad()) continue;
    int caxis = in->canonical_axis_index(axis_);
    int M = in->count(0, caxis);
    int N = in->count(caxis + 1, in->num_axes());
    int D = in->shape(caxis);
    cpu_softmax_backward<float>(M, D, N, out->const_data(), out->const_grad(), 1.0f, in->grad());
  }

}

}//namespace nn

}//namespace stensor