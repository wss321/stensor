/**
* Copyright 2021 wss
* Created by wss on 11月,24, 2021
*/
#include "linear_layer.hpp"

namespace stensor {

namespace nn {

void LinearLayer::backward_gpu() {
  for (int i = 0; i < inputs_.size(); ++i) {
    Tensor* x=inputs_[i];
    Tensor* y=outputs_[i];
    int caxis = x->canonical_axis_index(axis_);
    stensor::backward::matmul_backward(x, W_.get(), y);
    if (has_bias_){
      int M = y->count(0, caxis - 1);
      int N = y->count(caxis, y->num_axes());
      int D = y->shape(caxis - 1);
      stensor::gpu_reduce_sum(M, D, N,
                              y->const_grad(), 1.0f, b_->grad());
    }
  }
  inputs_.clear();
}

}
}