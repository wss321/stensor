/**
* Copyright 2021 wss
* Created by wss on 11月,24, 2021
*/
#include "linear_layer.hpp"

namespace stensor {

namespace nn {

void LinearLayer::backward_gpu() {
  for (int i = 0; i < inputs_.size(); ++i) {
    SharedTensor x(inputs_[i]);
    SharedTensor y(outputs_[i]);
    int caxis = x->canonical_axis_index(axis_);
    stensor::backward::matmul_backward(x.get(), W_.get(), y.get());
    if(has_bias_)
    stensor::gpu_reduce_sum(y->count(0, caxis),
                            y->shape(caxis),
                            y->count(caxis + 1, y->num_axes()),
                            y->const_grad(), 1.0f, b_->grad());
  }
  inputs_.clear();
}

}
}