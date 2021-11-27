/**
* Copyright 2021 wss
* Created by wss on 11月,26, 2021
*/
#ifndef STENSOR_OPTIMIZER_ADAGRAD_HPP_
#define STENSOR_OPTIMIZER_ADAGRAD_HPP_
#include "optimizer.hpp"

namespace stensor {

namespace optim {
class AdaGrad : public Optimizer {
 public:
  explicit AdaGrad(nn::Module *model,
                   float learning_rate = 1e-3,
                   float weight_decay = 0.0,
                   float eps = 1e-8f);
  ~AdaGrad() {};
  inline void step() {
    if (module_->state() == CPU) {
      step_cpu();
    } else step_gpu();
  }
 private:
  void step_cpu();
  void step_gpu();
  float eps_;
  nn::TensorVec history_;

 DISABLE_COPY_AND_ASSIGN(AdaGrad);
};
}//namespace optim
}//namespace stensor
#endif //STENSOR_OPTIMIZER_ADAGRAD_HPP_
