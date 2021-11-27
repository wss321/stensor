/**
* Copyright 2021 wss
* Created by wss on 11月,25, 2021
*/
#include "adagrad.hpp"
#include "core/math_tesnsor.hpp"
#include "math/math_base_cpu.hpp"
#include "math/math_base_cuda.hpp"

namespace stensor {

namespace optim {

AdaGrad::AdaGrad(nn::Module *model, float learning_rate, float weight_decay, float eps) {
  learning_rate_ = learning_rate;
  weight_decay_ = weight_decay;
  module_ = model;
  eps_ = eps;

  history_.clear();
  learnable_params_ = model->get_learnable_params();
  for (int i = 0; i < learnable_params_.size(); ++i) {
    nn::SharedTensor param = learnable_params_[i];
    Tensor *history = stensor::zeros(param->shape(), param->device(), false);
    history_.push_back(nn::SharedTensor(history));
  }
}

void AdaGrad::step_cpu() {
  if (weight_decay_ != 0)
    this->weight_decay();

  for (int i = 0; i < learnable_params_.size(); ++i) {
    nn::SharedTensor param = learnable_params_[i];
    nn::SharedTensor history_1 = history_[i];
    const float *g = param->const_grad();
    float *w = param->data();
    float *h = history_1->data();

    for (int j = 0; j < param->size(); ++j) {
      *h +=(*g)*(*g);
      *w -= learning_rate_ * ((*g) / (eps_ + std::sqrt(*h)));
      g++;
      w++;
      h++;
    }
  }
}

}//namespace optim
}//namespace stensor