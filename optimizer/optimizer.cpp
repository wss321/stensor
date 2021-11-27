/**
* Copyright 2021 wss
* Created by wss on 11月,25, 2021
*/
#include "optimizer.hpp"
#include "math/math_base_cpu.hpp"
#include "math/math_base_cuda.hpp"

namespace stensor {

namespace optim {
void Optimizer::weight_decay() {
  if (weight_decay_ != 0)
    switch (module_->state()) {
      case CPU:
        for (int i = 0; i < learnable_params_.size(); ++i) {
          nn::SharedTensor param = learnable_params_[i];
          cpu_axpy(param->size(), weight_decay_, param->const_data(), param->grad());
        }
        break;
      case GPU:
        for (int i = 0; i < learnable_params_.size(); ++i) {
          nn::SharedTensor param = learnable_params_[i];
          gpu_axpy(param->size(), weight_decay_, param->const_data(), param->grad());
        }
        break;
    }
}
}//namespace optim
}//namespace stensor