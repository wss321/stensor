/**
* Copyright 2021 wss
* Created by wss on 11月,25, 2021
*/
#ifndef STENSOR_OPTIMIZER_NESTEROV_HPP_
#define STENSOR_OPTIMIZER_NESTEROV_HPP_
#include "optimizer.hpp"

namespace stensor {

namespace optim {
class Nesterov : public Optimizer {
 public:
  explicit Nesterov(nn::Module *model,
               float learning_rate = 1e-3,
               float weight_decay = 0.0,
               float momentum = 0.0f);
  ~Nesterov() {};
  inline void step() {
    if (module_->state() == CPU) {
      step_cpu();
    } else step_gpu();
  }
 private:
  void step_cpu();
  void step_gpu();
  float momentum_;
  nn::TensorVec history_momentum_;

 DISABLE_COPY_AND_ASSIGN(Nesterov);
};
}//namespace optim
}//namespace stensor
#endif //STENSOR_OPTIMIZER_NESTEROV_HPP_
