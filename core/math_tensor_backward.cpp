/**
* Copyright 2021 wss
* Created by wss on 11月,21, 2021
*/
#include "math_tensor_backward.hpp"
namespace stensor {

namespace backward {

void add_backward(Tensor *a, Tensor *b, const Tensor *y) {
  std::vector<int> shape_a(a->shape());
  std::vector<int> shape_b(b->shape());
  std::vector<int> shape_out = stensor::broadcast(shape_a, shape_b);
  switch (a->state()) {
    case stensor::CPU:
      if (a->require_grad()) {
        if (shape_a == a->shape())
          cpu_copy(a->size(), y->const_grad(), a->grad());
        else {
          // TODO: broadcast add_backward
        }
      }
    case stensor::GPU:break;
  }
}
void matmul_backward(Tensor *a, Tensor *b, const Tensor *y,
                     int axis, bool transA, bool transB) {
  CHECK_EQ(a->device(), b->device()) << "tensors must be at same device";
  CHECK_EQ(y->device(), b->device()) << "tensors must be at same device";
  // inference shape
  int start_axis_a = a->canonical_axis_index(axis);
  int start_axis_b = b->canonical_axis_index(axis);
  int Ma = a->count(0, start_axis_a);
  int Na = a->count(start_axis_a, a->num_axes());
  if (transA) swap(Ma, Na);

  int Mb = b->count(0, start_axis_b);
  int Nb = b->count(start_axis_b, b->num_axes());
  if (transB) swap(Mb, Nb);

  std::vector<int> out_shape;
  for (int i = 0; i < start_axis_a; ++i)
    out_shape.push_back(a->shape(i));
  for (int i = start_axis_b; i < b->num_axes(); ++i)
    out_shape.push_back(b->shape(i));

  CHECK_EQ(Na, Mb) << "Shape mismatch";

  switch (a->state()) {
    case stensor::CPU:
      // D/a
      if (a->require_grad())
        stensor::cpu_gemm(false, !transB, Ma, Mb, Nb,
                          1.0f, y->const_grad(), b->const_data(),
                          1.0f, a->grad());
      // D/b
      if (b->require_grad())
        stensor::cpu_gemm(!transA, false, Mb, Nb, Ma,
                          1.0f, a->const_data(), y->const_grad(),
                          1.0f, b->grad());

      break;
    case stensor::GPU:
      // D/a
      if (a->require_grad())
        stensor::gpu_gemm(false, !transB, Ma, Mb, Nb,
                          1.0f, y->const_grad(), b->const_data(),
                          1.0f, a->grad());
      // D/b
      if (b->require_grad())
        stensor::gpu_gemm(!transA, false, Mb, Nb, Ma,
                          1.0f, a->const_data(), y->const_grad(),
                          1.0f, b->grad());
      break;
  }
}
}//namespace backward

}//namespace stensor
