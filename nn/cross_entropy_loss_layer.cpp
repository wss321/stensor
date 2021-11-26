/**
* Copyright 2021 wss
* Created by wss on 11月,24, 2021
*/
#include "cross_entropy_loss_layer.hpp"
#include "core/math_tensor_backward.hpp"
#include "math/math_base_cuda.hpp"
#include "core/transpose.hpp"
#include "core/math_tesnsor.hpp"

namespace stensor {

namespace nn {

CrossEntropyLossLayer::CrossEntropyLossLayer(const std::string &name, int axis, int device) : axis_(axis) {
  if (device > -1) state_ = GPU;
  else state_ = CPU;
  type_ = "CrossEntropyLoss";
  name_ = name;
  parameters_.resize(0);
  loss_ = 0;

}

TensorVec CrossEntropyLossLayer::forward(TensorVec &inputs) {
  CHECK_EQ(inputs.size(), 2) << "inputs must be two tensor:prediction & ground-truth";
  this->zero_output_grad();
  zero_temp_grad();
  inputs_ = inputs;
  SharedTensor pred = inputs_[0]; // float:e.g 64x100 (batch size:64, num class:100)
  SharedTensor gt = inputs_[1]; // int :64, 0~99

  int caxis = inputs[0]->canonical_axis_index(axis_);
  int M = pred->count(0, caxis);
  int N = pred->count(caxis + 1, pred->num_axes());
  int C = pred->shape(caxis);
  CHECK_EQ(M * N, gt->size());
  if (softmax_out_.get() == nullptr || pred->shape() != softmax_out_->shape()) {
    outputs_.resize(1);
    softmax_out_.reset(new Tensor(pred->shape(), pred->device(), pred->require_grad()));
    one_hot_.reset(new Tensor(pred->shape(), pred->device(), false));
    softmax_out_->set_name(name() + "/softmax_output");
    one_hot_->set_name(name() + "/one_hot");
    outputs_[0] = softmax_out_;
  }
  stensor::softmax(pred.get(), caxis, softmax_out_.get());// gpu err
  float *sm_data = softmax_out_->data();
  const float *gt_data = gt->const_data();
  loss_ = 0;
  for (int m = 0; m < M; ++m) {
    for (int c = 0; c < C; ++c) {
      for (int n = 0; n < N; ++n) {
        int gt_i = gt_data[n];
        CHECK_GE(gt_i, 0) << "ground truth must be great than 0 and less than number of classes";
        CHECK_LT(gt_i, C) << "ground truth must be great than 0 and less than number of classes";
        if (gt_i == c) loss_ -= log(*sm_data);
        sm_data++;
      }
    }
    gt_data += N;
  }
  loss_ /= M * N; // batch mean

  return outputs_;
}

void CrossEntropyLossLayer::backward_cpu() {
  SharedTensor in = inputs_[0];
  SharedTensor sm = softmax_out_;
  SharedTensor gt = inputs_[1];
  CHECK(in->shape_equal(sm.get()));

  if (!in->require_grad()) return;
  int caxis = in->canonical_axis_index(axis_);
  int num_class = in->shape(caxis);
  stensor::one_hot(gt.get(), num_class, one_hot_.get());

  const float *data_sm = sm->const_data();
  const float *data_oh = one_hot_->const_data();
  float *grad_in = in->grad();

  if (caxis != sm->num_axes() - 1) {
    std::vector<int> new_order;
    new_order.reserve(sm->num_axes());
    for (int i = 0; i < sm->num_axes(); ++i)
      new_order.push_back(i);
    new_order[sm->num_axes() - 1] = caxis;
    new_order[caxis] = sm->num_axes() - 1;
    Tensor *transpose_sm = stensor::transpose(sm.get(), new_order);//TODO inplace transpose
    data_sm = transpose_sm->const_data();
  }

  for (int i = 0; i < in->size(); ++i) {
    *grad_in += *data_sm - *data_oh;
    grad_in++;
    data_oh++;
    data_sm++;
  }
}

}//namespace nn

}//namespace stensor