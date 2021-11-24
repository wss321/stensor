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
  parameters_[0] = W_; // add parameter
  if (has_bias_) {
    Tensor *b = stensor::random({1, dim_out}, device, true);
    b_.reset(b);
    b_->set_name(name_ + "(" + type_ + ")" + "/bias");
    parameters_[1] = b_;// add parameter
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
void LinearLayer::backward_cpu() {
  for (int i = 0; i < inputs_.size(); ++i) {
    SharedTensor x(inputs_[i]);
    SharedTensor y(outputs_[i]);
    int caxis = x->canonical_axis_index(axis_);
    stensor::backward::matmul_backward(x.get(), W_.get(), y.get());
    if (has_bias_)
      stensor::cpu_reduce_sum(y->count(0, caxis),
                              y->shape(caxis),
                              y->count(caxis + 1, y->num_axes()),
                              y->const_grad(), 1.0f, b_->grad());
  }
  inputs_.clear();
}

void LinearLayer::backward() {
  if (LinearLayer::W_->state() == GPU)
    backward_gpu();
  else
    backward_cpu();
}

}//namespace nn

}//namespace stensor
