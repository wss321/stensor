/**
* Copyright 2021 wss
* Created by wss on 11月,25, 2021
*/
#include "sgd.hpp"
namespace stensor {

namespace optim {

SGD::SGD(nn::Module *model, float learning_rate, float weight_decay) {
  learning_rate_ = learning_rate;
  weight_decay_ = weight_decay;
  module_ = model;
  learnable_params_ = model->get_learnable_params();
}

void SGD::step_cpu() {
  //TODO:weight decay and momentum
  for (int i = 0; i < learnable_params_.size(); ++i) {
    nn::SharedTensor param = learnable_params_[i];
    cpu_axpy(param->size(), -learning_rate_, param->const_grad(), param->data());
  }
}

void SGD::step_gpu() {
  //TODO:weight decay and momentum
  for (int i = 0; i < learnable_params_.size(); ++i) {
    nn::SharedTensor param = learnable_params_[i];
    gpu_axpy(param->size(), -learning_rate_, param->const_grad(), param->data());
  }
}

}//namespace optim
}//namespace stensor