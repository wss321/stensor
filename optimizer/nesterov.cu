/**
* Copyright 2021 wss
* Created by wss on 11月,25, 2021
*/
#include "nesterov.hpp"
#include "math/math_base_cuda.hpp"

namespace stensor {

namespace optim {

template<typename Dtype>
__global__ void nesterov_kernel(const int n, const Dtype momentum,
                                const Dtype lr, const Dtype *grad, Dtype *v, Dtype *w) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype g = grad[index];
    v[index] = momentum * v[index] + g;
    w[index] -= lr * (momentum * v[index] + g);
  }
}

void Nesterov::step_gpu() {
  if (weight_decay_ != 0)
    this->weight_decay();
  for (int i = 0; i < learnable_params_.size(); ++i) {
    nn::SharedTensor param = learnable_params_[i];
    nn::SharedTensor history_mmt = history_momentum_[i];
    nesterov_kernel<float><<<GET_BLOCKS(param->size()), CUDA_NUM_THREADS>>>
    (param->size(), momentum_, learning_rate_, param->const_grad(), history_mmt->data(), param->data());
  }
  // original implement
//  for (int i = 0; i < learnable_params_.size(); ++i) {
//    nn::SharedTensor param = learnable_params_[i];
//    nn::SharedTensor history_mmt = history_momentum_[i];
//    gpu_axpby(history_mmt->size(), 1.0f, param->const_grad(), momentum_, history_mmt->data());
//    gpu_axpy(param->size(), momentum_, history_mmt->const_data(), param->grad());
//    gpu_axpy(param->size(), -learning_rate_, param->grad(), param->data());
//  }
}

}//namespace optim
}//namespace stensor