/**
* Copyright 2021 wss
* Created by wss on 11月,25, 2021
*/
#ifndef STENSOR_OPTIMIZER_OPTIMIZER_HPP_
#define STENSOR_OPTIMIZER_OPTIMIZER_HPP_
#include <vector>
#include <memory>
#include "module.hpp"

namespace stensor {

namespace optim {
class Optimizer {
 public:
  Optimizer() {};
  virtual ~Optimizer() = default;;
  virtual void step() = 0;
  inline virtual void zero_grad() {
    for (int i = 0; i < learnable_params_.size(); ++i)
      learnable_params_[i]->zero_grad();
  };
  inline void set_learning_rate(float lr) { learning_rate_ = lr; };
 protected:
  float learning_rate_;
  float weight_decay_;
  nn::Module* module_;
  nn::TensorVec learnable_params_;
};
}//namespace optim
}//namespace stensor
#endif //STENSOR_OPTIMIZER_OPTIMIZER_HPP_
