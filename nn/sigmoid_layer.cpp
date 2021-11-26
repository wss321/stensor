/**
* Copyright 2021 wss
* Created by wss on 11月,26, 2021
*/
#include "sigmoid_layer.hpp"
#include "core/math_tesnsor.hpp"

namespace stensor {

namespace nn {
TensorVec Sigmoid::forward(TensorVec &inputs) {
  CHECK_EQ(inputs.size(), 1) << "Only support one input tensor now";
  this->zero_output_grad();
  SharedTensor in = inputs[0];
  if (result_.get() == nullptr || in->shape() != result_->shape()) {
    outputs_.resize(1);
    result_.reset(new Tensor(in->shape(), in->device(), in->require_grad()));
    result_->set_name(name() + "/output");
  }
  inputs_ = inputs;
  stensor::tanh(in.get(), result_.get());
  if (outputs_.empty())
    outputs_.push_back(result_);
  else outputs_[0] = result_;
  return outputs_;
}

void Sigmoid::backward_cpu() {
  SharedTensor x(inputs_[0]);
  SharedTensor y(outputs_[0]);
  const float *y_grad = y->const_grad();
  const float *y_data = y->const_data();
  float *x_grad = x->grad();

  for (int j = 0; j < x->size(); ++j) {
    *x_grad = (*y_grad) * (*y_data) * (1 - (*y_data));
    y_data++;
    x_grad++;
    y_grad++;
  }

  inputs_.clear();
}

void Sigmoid::backward() {
  if (state_ == GPU)
    backward_gpu();
  else
    backward_cpu();
}

}//namespace nn

}//namespace stensor