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
  if (device > -1) state_ = GPU;
  else state_ = CPU;
  has_bias_ = bias;
  parameters_.resize(2);
  Tensor *W = stensor::random_gaussian({dim_in, dim_out}, 0, 0.1, device, true);
  W->set_name(name_ + "(" + type_ + ")" + "/W");
  W_.reset(W);
  parameters_[0] = W_; // add parameter
  if (has_bias_) {
//    Tensor *b = stensor::random({1, dim_out}, -1, 1, device, true);
    Tensor *b = stensor::zeros({1, dim_out}, device, true);
    b_.reset(b);
    b_->set_name(name_ + "(" + type_ + ")" + "/bias");
    parameters_[1] = b_;// add parameter
  }
}

//TODO:fix matmul (a,b,c) err
std::vector<Tensor *> LinearLayer::forward(std::vector<Tensor *> &inputs) {
  CHECK_EQ(inputs.size(), 1) << "Only support one input tensor now";
  Tensor *in = inputs[0];
  std::vector<int> out_shape(in->shape());
  out_shape[in->canonical_axis_index(axis_)] = W_->shape(-1);
  if (result_.get() == nullptr || out_shape != result_->shape()) {
    outputs_.resize(1);
    result_.reset(new Tensor(out_shape, in->device(), W_->require_grad() || in->require_grad()));
    result_->set_name(name()+"/output");
    outputs_[0] = result_.get();
  }
//  std::cout<<outputs_[0].get();
  inputs_ = inputs;
  Tensor* m=stensor::matmul(in, W_.get(), axis_);
//  stensor::matmul(in.get(), W_.get(), axis_,
//                  false, false, 0.0, result_.get());

  if (has_bias_)
    stensor::add(m, b_.get(), m);

  if (outputs_.empty())
    outputs_.push_back(m);
  else outputs_[0] = m;
  result_.reset(m);
  return outputs_;

//  if (result_.get() == nullptr || out_shape != result_->shape()) {
//    outputs_.resize(1);
//    result_.reset(new Tensor(out_shape, in->device(), W_->require_grad() || in->require_grad()));
//    result_->set_name(name() + "/output");
//    outputs_[0] = result_.get();
//  }
////  std::cout<<outputs_[0].use_count()<<"\n";
//
//  inputs_ = inputs;
////  SharedTensor m(stensor::matmul(in.get(), W_.get(), axis_));
//  std::cout << result_.get();
//  stensor::matmul(in, W_.get(), axis_,
//                  false, false, 0.0, result_.get());
//
//  if (has_bias_)
//    stensor::add(result_.get(), b_.get(), result_.get());
//
//  return std::vector<Tensor *>({result_.get()});
}
void LinearLayer::backward_cpu() {
  for (int i = 0; i < inputs_.size(); ++i) {
    Tensor *x = inputs_[0];
    int caxis = x->canonical_axis_index(axis_);
    stensor::backward::matmul_backward(x, W_.get(), result_.get());
    if (has_bias_) {
      int M = result_->count(0, caxis - 1);
      int N = result_->count(caxis, result_->num_axes());
      int D = result_->shape(caxis - 1);
      stensor::cpu_reduce_sum(M, D, N,
                              result_->const_grad(), 1.0f, b_->grad());
    }

  }
  inputs_.clear();
}

void LinearLayer::backward() {
  if (state_ == GPU)
    backward_gpu();
  else
    backward_cpu();
}

}//namespace nn

}//namespace stensor
