/**
* Copyright 2021 wss
* Created by wss on 11月,25, 2021
*/
#include "adam.hpp"
#include "core/math_tesnsor.hpp"
#include "math/math_base_cpu.hpp"
#include "math/math_base_cuda.hpp"

namespace stensor {

namespace optim {

Adam::Adam(nn::Module *model, float learning_rate, float weight_decay, float beta1,
           float beta2, float eps) {
  learning_rate_ = learning_rate;
  weight_decay_ = weight_decay;
  module_ = model;
  beta1_ = beta1;
  beta2_ = beta2;
  eps_ = eps;
  iter_ = 0;

  history_1order_.clear();
  history_2order_.clear();
  denom_.clear();
  learnable_params_ = model->get_learnable_params();
  for (int i = 0; i < learnable_params_.size(); ++i) {
    nn::SharedTensor param = learnable_params_[i];
    Tensor *history_1 = stensor::zeros(param->shape(), param->device(), false);
    Tensor *history_2 = stensor::zeros(param->shape(), param->device(), false);
    Tensor *denom = stensor::zeros(param->shape(), param->device(), false);
    history_1order_.push_back(nn::SharedTensor(history_1));
    history_2order_.push_back(nn::SharedTensor(history_2));
    denom_.push_back(nn::SharedTensor(denom));
  }
}
//TODO:accelerate using cblas
//void Adam::step_cpu() {
//  if (weight_decay_ != 0)
//    this->weight_decay();
//  iter_++;
//  float bf1 = 1.0 / (1.0 - std::pow(beta1_, iter_));
//  float bf2 = 1.0 / (1.0 - std::pow(beta2_, iter_));
//  for (int i = 0; i < learnable_params_.size(); ++i) {
//    nn::SharedTensor param = learnable_params_[i];
//    nn::SharedTensor history_1 = history_1order_[i];
//    nn::SharedTensor history_2 = history_2order_[i];
//    const float *g = param->const_grad();
//    float *w = param->data();
//    float *h1 = history_1->data();
//    float *h2 = history_2->data();
//
//    for (int j = 0; j < param->size(); ++j) {
//      *h1 = beta1_ * (*h1) + (1 - beta1_) * (*g);
//      *h2 = beta2_ * (*h2) + (1 - beta2_) * (*g)*(*g);
//      float m_hat = *h1 * bf1;
//      float v_hat = *h2 * bf2;
//      *w -= learning_rate_ * (m_hat / (eps_ + std::sqrt(v_hat)));
//      g++;
//      w++;
//      h1++;
//      h2++;
//    }
//  }
//}

void Adam::step_cpu() {
  if (weight_decay_ != 0)
    this->weight_decay();
  iter_++;
  float bf1 = 1.0 / (1.0 - std::pow(beta1_, iter_));
  float bf2 = 1.0 / (1.0 - std::pow(beta2_, iter_));
  for (int i = 0; i < learnable_params_.size(); ++i) {
    nn::SharedTensor param = learnable_params_[i];
    nn::SharedTensor history_1 = history_1order_[i];
    nn::SharedTensor history_2 = history_2order_[i];
    nn::SharedTensor denom = denom_[i];
    int n = param->size();
    const float *g = param->const_grad();
    float *w = param->data();
    float *h1 = history_1->data();
    float *h2 = history_2->data();
    float *d = denom->data();

    cpu_axpby(n, 1 - beta1_, g, beta1_, h1);
    cpu_mul(n, g, g, d); // g^2
    cpu_axpby(n, 1 - beta2_, d, beta2_, h2);
    cpu_scale(n, h2, bf2, d);
    cpu_sqrt(n, d, d);
    cpu_add_scalar(n, d, eps_, d);
    cpu_div(n, h1, d, d);
    cpu_axpy(n, -learning_rate_ * bf1, d, w);
  }
}

}//namespace optim
}//namespace stensor