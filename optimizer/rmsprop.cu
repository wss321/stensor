/**
* Copyright 2021 wss
* Created by wss on 11月,25, 2021
*/
#include "rmsprop.hpp"
#include "math/math_base_cpu.hpp"
#include "math/math_base_cuda.hpp"

namespace stensor {

namespace optim {

template<typename Dtype>
__global__ void adam_kernel(const int n, const Dtype beta,
                            const Dtype lr, const Dtype eps,
                            const Dtype *grad, Dtype *h2, Dtype *w) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype g = grad[index];
    h2[index] = beta * h2[index] + (1 - beta) * g*g;

    w[index] -= lr * (g / (eps + sqrt(h2[index])));
  }
}

void RMSprop::step_gpu() {
  if (weight_decay_ != 0)
    this->weight_decay();

  for (int i = 0; i < learnable_params_.size(); ++i) {
    nn::SharedTensor param = learnable_params_[i];
    nn::SharedTensor history_2 = history_2order_[i];
    adam_kernel<float><<<GET_BLOCKS(param->size()), CUDA_NUM_THREADS>>>
        (param->size(), beta_, learning_rate_,
         eps_, param->grad(), history_2->data(), param->data());
  }

}

}//namespace optim
}//namespace stensor