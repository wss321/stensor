/**
* Copyright 2021 wss
* Created by wss on 11月,21, 2021
*/
#include "linear_layer.hpp"
#include "core/math_tensor_backward.hpp"
#include "math/math_base_cuda.hpp"
#include "math/math_base_cpu.hpp"
#include "core/math_tesnsor.hpp"

namespace stensor {

namespace nn {

Linear::Linear(const std::string &name,
               int dim_in,
               int dim_out,
               int axis,
               int device,
               bool bias) {
  CHECK_GT(dim_in, 0);
  CHECK_GT(dim_out, 0);
  this->type_ = "Linear";
  this->name_ = name;
  if (device > -1) this->state_ = GPU;
  else this->state_ = CPU;
  axis_ = axis;
  has_bias_ = bias;
  if (has_bias_)
    this->parameters_.resize(2);
  else this->parameters_.resize(1);
  Tensor *W = stensor::random_gaussian({dim_in, dim_out}, 0, 0.1, device, true);
  W->set_name(name_ + "(" + type_ + ")" + "/W");
  W_.reset(W);
  this->parameters_[0] = W_; // add parameter
  if (has_bias_) {
//    Tensor *b = stensor::random({1, dim_out}, -1, 1, device, true);
    Tensor *b = stensor::zeros({1, dim_out}, device, true);
    b_.reset(b);
    b_->set_name(name_ + "(" + type_ + ")" + "/bias");
    this->parameters_[1] = b_;// add parameter
  }
}

TensorVec Linear::forward(TensorVec &inputs) {
  CHECK_EQ(inputs.size(), 1) << "Only support one input tensor now";
  this->zero_output_grad();
  SharedTensor in = inputs[0];
  std::vector<int> out_shape(in->shape());
  out_shape[in->canonical_axis_index(axis_)] = W_->shape(-1);
  if (result_.get() == nullptr || out_shape != result_->shape()) {
    outputs_.resize(1);
    result_.reset(new Tensor(out_shape, in->device(), W_->require_grad() || in->require_grad()));
    result_->set_name(name() + "/output");
  }
  inputs_ = inputs;
  stensor::matmul(in.get(), W_.get(), axis_,
                  false, false, 0.0, result_.get());

  if (has_bias_)
    stensor::add(result_.get(), b_.get(), result_.get());

  if (outputs_.empty())
    outputs_.push_back(result_);
  else outputs_[0] = result_;
  return outputs_;
}
void Linear::backward_cpu() {
  for (int i = 0; i < inputs_.size(); ++i) {
    SharedTensor x(inputs_[i]);
    SharedTensor y(outputs_[i]);
    int caxis = x->canonical_axis_index(axis_);
    stensor::backward::matmul_backward(x.get(), W_.get(), y.get());
    if (has_bias_) {
      int M = y->count(0, caxis - 1);
      int N = y->count(caxis, y->num_axes());
      int D = y->shape(caxis - 1);
      stensor::cpu_reduce_sum(M, D, N,
                              y->const_grad(), 1.0f, b_->grad());
    }

  }
  inputs_.clear();
}

void Linear::backward() {
  if (state_ == GPU)
    backward_gpu();
  else
    backward_cpu();
}

}//namespace nn

}//namespace stensor
