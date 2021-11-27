/**
* Copyright 2021 wss
* Created by wss on 11月,25, 2021
*/
#ifndef STENSOR_OPTIMIZER_RMSPROP_HPP_
#define STENSOR_OPTIMIZER_RMSPROP_HPP_
#include "optimizer.hpp"

namespace stensor {

namespace optim {
class RMSprop : public Optimizer {
 public:
  explicit RMSprop(nn::Module *model,
                float learning_rate = 1e-3,
                float weight_decay = 0.0,
                float beta = 0.9f,
                float eps = 1e-8f);
  ~RMSprop() {};
  inline void step() {
    if (module_->state() == CPU) {
      step_cpu();
    } else step_gpu();
  }
 private:
  void step_cpu();
  void step_gpu();
  float beta_;
  float eps_;
  nn::TensorVec history_2order_;
  nn::TensorVec denom_;

 DISABLE_COPY_AND_ASSIGN(RMSprop);
};
}//namespace optim
}//namespace stensor
#endif //STENSOR_OPTIMIZER_RMSPROP_HPP_
