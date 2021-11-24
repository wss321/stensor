/**
* Copyright 2021 wss
* Created by wss on 11月,21, 2021
*/
#include "linear_layer.hpp"

namespace stensor {

namespace nn {

LinearLayer::LinearLayer(const std::string &name,
                         int dim_in,
                         int dim_out,
                         int axis,
                         int device,
                         bool bias) {
  CHECK_GT(dim_in, 0);
  CHECK_GT(dim_out, 0);
  type_ = "Linear";
  name_ = name;
  axis_ = axis;
  has_bias_ = bias;
  parameters_.resize(2);
  Tensor *W = stensor::random({dim_in, dim_out}, device, true);
  W->set_name(name_ + "(" + type_ + ")" + "/W");
  W_.reset(W);
  parameters_[0] = W_;
  if (has_bias_) {
    Tensor *b = stensor::random({1, dim_out}, device, true);
    b_.reset(b);
    b_->set_name(name_ + "(" + type_ + ")" + "/bias");
    parameters_[1] = b_;
  }

}

TensorVec LinearLayer::forward(TensorVec &inputs) {
  outputs_.resize(inputs.size());
  inputs_ = inputs;
  Tensor *m;
  Tensor *m1;
  for (int i = 0; i < inputs.size(); ++i) {
    m = stensor::matmul(inputs[i].get(), W_.get(), axis_);
    if (has_bias_) {
      m1 = stensor::add(m, b_.get());
      delete m;
      m = m1;
    }
    outputs_[i].reset(m);
  }
  return outputs_;
}

void LinearLayer::backward() {
  for (int i = 0; i < inputs_.size(); ++i) {

  }
  inputs_.clear();
}

}//namespace nn

}//namespace stensor
