/**
* Copyright 2021 wss
* Created by wss on 11月,25, 2021
*/
#include "sgd.hpp"
#include "math/math_base_cpu.hpp"
#include "math/math_base_cuda.hpp"

namespace stensor {

namespace optim {

SGD::SGD(nn::Module *model, float learning_rate, float weight_decay, float momentum) {
  learning_rate_ = learning_rate;
  weight_decay_ = weight_decay;
  momentum_ = momentum;
  module_ = model;
  history_momentum_.clear();
  learnable_params_ = model->get_learnable_params();
  for (int i = 0; i < learnable_params_.size(); ++i) {
    nn::SharedTensor param = learnable_params_[i];
    Tensor *history_mmt = new Tensor(param->shape(), param->device(), false);
    history_momentum_.push_back(nn::SharedTensor(history_mmt));
  }
}

void SGD::step_cpu() {
  if (weight_decay_!=0)
    this->weight_decay();
  if (momentum_==0)
    for (int i = 0; i < learnable_params_.size(); ++i) {
      nn::SharedTensor param = learnable_params_[i];
      cpu_axpy(param->size(), -learning_rate_, param->const_grad(), param->data());
    }
  else
    for (int i = 0; i < learnable_params_.size(); ++i) {
      nn::SharedTensor param = learnable_params_[i];
      nn::SharedTensor history_mmt = history_momentum_[i];
      cpu_axpby(history_mmt->size(), 1.0f, param->const_grad(), momentum_, history_mmt->data());
      cpu_axpy(param->size(), -learning_rate_, history_mmt->const_data(), param->data());
    }
}

void SGD::step_gpu() {
  if (weight_decay_!=0)
    this->weight_decay();
  if (momentum_==0)
    for (int i = 0; i < learnable_params_.size(); ++i) {
      nn::SharedTensor param = learnable_params_[i];
      gpu_axpy(param->size(), -learning_rate_, param->const_grad(), param->data());
    }
  else
    for (int i = 0; i < learnable_params_.size(); ++i) {
      nn::SharedTensor param = learnable_params_[i];
      nn::SharedTensor history_mmt = history_momentum_[i];
      gpu_axpby(history_mmt->size(), 1.0f, param->const_grad(), momentum_, history_mmt->data());
      gpu_axpy(param->size(), -learning_rate_, history_mmt->const_data(), param->data());
    }
}

}//namespace optim
}//namespace stensor