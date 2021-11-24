/**
* Copyright 2021 wss
* Created by wss on 11月,24, 2021
*/
#include "cross_entropy_loss_layer.hpp"

namespace stensor {

namespace nn {
template<typename Dtype>
__global__ void cross_entropy_backward_kernel(const int n,
                                              const Dtype *sm_data,
                                              const Dtype *onehot_data,
                                              Dtype beta,
                                              Dtype *x_grad) {
  CUDA_KERNEL_LOOP(index, n) {
    x_grad[index] = sm_data[index] - onehot_data[index] + beta * x_grad[index];
  }
}

void CrossEntropyLossLayer::backward_gpu() {
  SharedTensor in = inputs_[0];
  SharedTensor sm = softmax_out_;
  SharedTensor gt = inputs_[1];
  CHECK(in->shape_equal(sm.get()));
  if (!in->require_grad()) return;
  int caxis = in->canonical_axis_index(axis_);
  int num_class = in->shape(caxis);
  Tensor *one_hot = stensor::one_hot(gt.get(), num_class);
  one_hot_.reset(one_hot);
  const float *data_sm = sm->const_data();
  const float *data_gt = one_hot->const_data();
  float *grad_in = in->grad();

  if (caxis != sm->num_axes() - 1) {
    std::vector<int> new_order;
    new_order.reserve(sm->num_axes());
    for (int i = 0; i < sm->num_axes(); ++i)
      new_order.push_back(i);
    new_order[sm->num_axes() - 1] = caxis;
    new_order[caxis] = sm->num_axes() - 1;
    Tensor *transpose_sm = stensor::transpose(sm.get(), new_order);
    data_sm = transpose_sm->const_data();
  }
  cross_entropy_backward_kernel<float>
  <<<GET_BLOCKS(in->size()), CUDA_NUM_THREADS>>>(in->size(), data_sm, data_gt, 1.0f, grad_in);
  cudaDeviceSynchronize();
  CUDA_POST_KERNEL_CHECK;

}
}//namespace nn

}//namespace stensor