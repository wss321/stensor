/**
* Copyright 2021 wss
* Created by wss on 11月,25, 2021
*/
#include "adam.hpp"
#include "math/math_base_cpu.hpp"
#include "math/math_base_cuda.hpp"

namespace stensor {

namespace optim {

template<typename Dtype>
__global__ void adam_kernel(const int n, const Dtype beta1, const Dtype beta2,
                            const Dtype c1, const Dtype c2,
                            const Dtype lr, const Dtype eps,
                            const Dtype *grad, Dtype *h1, Dtype *h2, Dtype *w) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype g = grad[index];
    // m_t=\beta_1 m_{t-1}+(1-\beta_1) \Delta w_t
    h1[index] = beta1 * h1[index] + (1 - beta1) * g;
    // v_t=\beta_2 v_{t-1}+(1-\beta_2) (\Delta w_t)^2
    h2[index] = beta2 * h2[index] + (1 - beta2) * g * g;
    // \hat{m_t}=\frac{m_t}{1-\beta_1^t}
    Dtype m_hat = h1[index] * c1;
    // \hat{v_t}=\frac{v_t}{1-\beta_2^t}
    Dtype v_hat = h2[index] * c2;
    //w_{t+1}=w_t-\eta \frac{\hat{m_t}}{\sqrt{\hat{v_t}}+\epsilon}
    w[index] -= lr * (m_hat / (eps + sqrt(v_hat)));
  }
}

void Adam::step_gpu() {
  if (weight_decay_ != 0)
    this->weight_decay();
  iter_++;
  float bias_correction1 = 1.0 / (1.0 - pow(beta1_, iter_));
  float bias_correction2 = 1.0 / (1.0 - pow(beta2_, iter_));
  for (int i = 0; i < learnable_params_.size(); ++i) {
    nn::SharedTensor param = learnable_params_[i];
    nn::SharedTensor history_1 = history_1order_[i];
    nn::SharedTensor history_2 = history_2order_[i];
    adam_kernel<float><<<GET_BLOCKS(param->size()), CUDA_NUM_THREADS>>>
        (param->size(), beta1_, beta2_, bias_correction1, bias_correction2, learning_rate_,
         eps_, param->grad(), history_1->data(), history_2->data(), param->data());
  }

}

}//namespace optim
}//namespace stensor