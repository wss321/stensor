/**
* Copyright 2021 wss
* Created by wss on 11月,25, 2021
*/
#include "rmsprop.hpp"
#include "core/math_tesnsor.hpp"
#include "math/math_base_cpu.hpp"
#include "math/math_base_cuda.hpp"

namespace stensor {

namespace optim {

RMSprop::RMSprop(nn::Module *model, float learning_rate, float weight_decay, float beta, float eps) {
  learning_rate_ = learning_rate;
  weight_decay_ = weight_decay;
  module_ = model;
  beta_ = beta;
  eps_ = eps;

  history_2order_.clear();
  denom_.clear();
  learnable_params_ = model->get_learnable_params();
  for (int i = 0; i < learnable_params_.size(); ++i) {
    nn::SharedTensor param = learnable_params_[i];
    Tensor *history_2 = stensor::zeros(param->shape(), param->device(), false);
    Tensor *denom = stensor::zeros(param->shape(), param->device(), false);
    history_2order_.push_back(nn::SharedTensor(history_2));
    denom_.push_back(nn::SharedTensor(denom));
  }
}
//TODO:accelerate using cblas
void RMSprop::step_cpu() {
  if (weight_decay_ != 0)
    this->weight_decay();
  for (int i = 0; i < learnable_params_.size(); ++i) {
    nn::SharedTensor param = learnable_params_[i];
    nn::SharedTensor history_2 = history_2order_[i];
    const float *g = param->const_grad();
    float *w = param->data();
    float *h2 = history_2->data();

    for (int j = 0; j < param->size(); ++j) {
      *h2 = beta_ * (*h2) + (1 - beta_) * (*g)*(*g);
      *w -= learning_rate_ * (*g / (eps_ + std::sqrt(*h2)));
      g++;
      w++;
      h2++;
    }
  }
}

//void RMSprop::step_cpu() {
//  if (weight_decay_ != 0)
//    this->weight_decay();
//  for (int i = 0; i < learnable_params_.size(); ++i) {
//    nn::SharedTensor param = learnable_params_[i];
//    nn::SharedTensor history_2 = history_2order_[i];
//    nn::SharedTensor denom = denom_[i];
//    int n = param->size();
//    const float *g = param->const_grad();
//    float *w = param->data();
//    float *h2 = history_2->data();
//    float *d = denom->data();
//
//    cpu_mul(n, g, g, d); // g^2
//    cpu_axpby(n, 1 - beta_, d, beta_, h2);
//    cpu_sqrt(n, d, d);
//    cpu_add_scalar(n, d, eps_, d);
//    cpu_div(n, g, d, d);
//    cpu_axpy(param->size(), -learning_rate_, d, w);
//  }
//}

}//namespace optim
}//namespace stensor