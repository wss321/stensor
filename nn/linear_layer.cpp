/**
* Copyright 2021 wss
* Created by wss on 11月,21, 2021
*/
#include "linear_layer.hpp"

namespace stensor {

namespace nn {
LinearLayer::LinearLayer(int dim_in, int dim_out, int axis, int device, bool bias) {
  CHECK_GT(dim_in, 0);
  CHECK_GT(dim_out, 0);
  axis_ = axis;
  has_bias_ = bias;
  Tensor *W = stensor::random({dim_in, dim_out}, device, true);
  W->set_name("weight");
  if (has_bias_) {
    Tensor *b = stensor::random({1, dim_out}, device, true);
    b_.reset(b);
  }
  W_.reset(W);

}

TensorVec LinearLayer::forward(TensorVec &inputs) {
  TensorVec output(inputs.size());
  Tensor *m;
  Tensor *m1;
  for (int i = 0; i < inputs.size(); ++i) {
    m = stensor::matmul(inputs[i].get(), W_.get());
    if (has_bias_) {
      m1 = stensor::add(m, b_.get());
      delete m;
      m = m1;
    }
    output[i].reset(m);
    bottom_.push_back(inputs[i]);
    top_.push_back(output[i]);
  }

  return output;
}

void LinearLayer::backward() {
  for (int i = 0; i < bottom_.size(); ++i) {

  }
  bottom_.clear();
}

}//namespace nn

}//namespace stensor
