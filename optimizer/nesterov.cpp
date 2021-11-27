/**
* Copyright 2021 wss
* Created by wss on 11月,25, 2021
*/
#include "nesterov.hpp"
#include "core/math_tesnsor.hpp"
#include "math/math_base_cpu.hpp"

namespace stensor {

namespace optim {

Nesterov::Nesterov(nn::Module *model, float learning_rate, float weight_decay, float momentum) {
  learning_rate_ = learning_rate;
  weight_decay_ = weight_decay;
  momentum_ = momentum;
  module_ = model;
  history_momentum_.clear();
  learnable_params_ = model->get_learnable_params();
  for (int i = 0; i < learnable_params_.size(); ++i) {
    nn::SharedTensor param = learnable_params_[i];
    Tensor *history_mmt = stensor::zeros(param->shape(), param->device(), false);
    history_momentum_.push_back(nn::SharedTensor(history_mmt));
  }
}

void Nesterov::step_cpu() {
  if (weight_decay_ != 0)
    this->weight_decay();
  for (int i = 0; i < learnable_params_.size(); ++i) {
    nn::SharedTensor param = learnable_params_[i];
    nn::SharedTensor history_mmt = history_momentum_[i];
    cpu_axpby(history_mmt->size(), 1.0f, param->const_grad(), momentum_, history_mmt->data());
    cpu_axpy(param->size(), momentum_, history_mmt->const_data(), param->grad());
    cpu_axpy(param->size(), -learning_rate_, param->grad(), param->data());
  }

//  for (int i = 0; i < learnable_params_.size(); ++i) {
//    nn::SharedTensor param = learnable_params_[i];
//    nn::SharedTensor history_mmt = history_momentum_[i];
//    const float *g = param->const_grad();
//    float *w = param->data();
//    float *h = history_mmt->data();
//    for (int j = 0; j < param->size(); ++j) {
//      *h = momentum_ * (*h) + (*g);
//      *w -= learning_rate_ * (momentum_ * (*h) + (*g));
//      g++;
//      w++;
//      h++;
//    }
//  }
}

}//namespace optim
}//namespace stensor