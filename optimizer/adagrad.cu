/**
* Copyright 2021 wss
* Created by wss on 11月,25, 2021
*/
#include "adagrad.hpp"
#include "math/math_base_cpu.hpp"
#include "math/math_base_cuda.hpp"

namespace stensor {

namespace optim {

template<typename Dtype>
__global__ void rmsprop_kernel(const int n, const Dtype eps,
                               const Dtype lr, const Dtype *grad, Dtype *v, Dtype *w) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype g = grad[index];
    v[index] += g * g;
    w[index] -= lr * (g / (eps + sqrt(v[index])));
  }
}
void AdaGrad::step_gpu() {
  if (weight_decay_ != 0)
    this->weight_decay();
  for (int i = 0; i < learnable_params_.size(); ++i) {
    nn::SharedTensor param = learnable_params_[i];
    nn::SharedTensor history_1 = history_[i];
    rmsprop_kernel<float><<<GET_BLOCKS(param->size()), CUDA_NUM_THREADS>>>
        (param->size(), eps_, learning_rate_,
         param->grad(), history_1->data(), param->data());
  }

}

}//namespace optim
}//namespace stensor