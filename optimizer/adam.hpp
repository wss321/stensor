/**
* Copyright 2021 wss
* Created by wss on 11月,25, 2021
*/
#ifndef STENSOR_OPTIMIZER_ADAM_HPP_
#define STENSOR_OPTIMIZER_ADAM_HPP_
#include "optimizer.hpp"

namespace stensor {

namespace optim {
class Adam : public Optimizer {
 public:
  explicit Adam(nn::Module *model,
                float learning_rate = 1e-3,
                float weight_decay = 0.0,
                float beta1 = 0.9f,
                float beta2 = 0.999f,
                float eps = 1e-8f);
  ~Adam() {};
  inline void step() {
    if (module_->state() == CPU) {
      step_cpu();
    } else step_gpu();
  }
 private:
  void step_cpu();
  void step_gpu();
  float beta1_;
  float beta2_;
  float eps_;
  int iter_;
  nn::TensorVec history_1order_;
  nn::TensorVec history_2order_;
  nn::TensorVec denom_;

 DISABLE_COPY_AND_ASSIGN(Adam);
};
}//namespace optim
}//namespace stensor
#endif //STENSOR_OPTIMIZER_ADAM_HPP_
