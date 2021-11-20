/**
* Copyright 2021 wss
* Created by wss on 11月,16, 2021
*/
#include "math_tesnsor.hpp"

namespace stensor {
/* math of Tensor */
/* self-op start*/

Tensor *sigmoid(Tensor *tensor, bool inplace) {
  float *in_data = tensor->data();
  float *out_data = tensor->data();
  Tensor *out_tensor = nullptr;
  if (!inplace) {
    out_tensor = new Tensor(tensor, tensor->require_grad());
    out_data = out_tensor->data();
  }
  switch (tensor->state()) {
    case stensor::CPU:stensor::cpu_sigmoid(tensor->size(), in_data, out_data);
      break;
    case stensor::GPU:stensor::gpu_sigmoid(tensor->size(), in_data, out_data);
      break;
  }
  return out_tensor;
}
#define IMPLEMENT_TENSOR_UNARY_FUNC(name) \
Tensor *name(Tensor *tensor, bool inplace) {\
  float *in_data = tensor->data();\
  float *out_data = tensor->data();\
  Tensor *out_tensor = nullptr;\
  if (!inplace) {\
    out_tensor = new Tensor(tensor, tensor->require_grad());\
    out_data = out_tensor->data();\
  }\
  switch (tensor->state()) {\
    case stensor::CPU:stensor::cpu_##name(tensor->size(), in_data, out_data);\
      break;\
    case stensor::GPU:stensor::gpu_##name(tensor->size(), in_data, out_data);\
      break;\
  }\
  return out_tensor;\
}

IMPLEMENT_TENSOR_UNARY_FUNC(tanh);
IMPLEMENT_TENSOR_UNARY_FUNC(relu);
IMPLEMENT_TENSOR_UNARY_FUNC(elu);
IMPLEMENT_TENSOR_UNARY_FUNC(gelu);
IMPLEMENT_TENSOR_UNARY_FUNC(leakyrelu);
IMPLEMENT_TENSOR_UNARY_FUNC(sign);
IMPLEMENT_TENSOR_UNARY_FUNC(sqrt);
IMPLEMENT_TENSOR_UNARY_FUNC(square);

Tensor *clamp(Tensor *tensor, float minVal, float maxVal, bool inplace) {
  float *in_data = tensor->data();
  float *out_data = tensor->data();
  Tensor *out_tensor = nullptr;
  if (!inplace) {
    out_tensor = new Tensor(tensor, tensor->require_grad());
    out_data = out_tensor->data();
  }
  switch (tensor->state()) {
    case stensor::CPU:stensor::cpu_clamp(out_tensor->size(), minVal, maxVal, in_data, out_data);
      break;
    case stensor::GPU:stensor::gpu_clamp(out_tensor->size(), minVal, maxVal, in_data, out_data);
      break;
  }
  return out_tensor;
}

/* self-op end*/

void set(Tensor *tensor, float val) {
  Tensor::Dtype *data = tensor->data();
  switch (tensor->state()) {
    case stensor::CPU:stensor::cpu_set(tensor->size(), val, data);
      break;
    case stensor::GPU:stensor::gpu_set(tensor->size(), val, data);
      break;
  }
}

Tensor *add(Tensor *tensor, float val, bool inplace) {
  float *in_data = tensor->data();
  float *out_data = tensor->data();
  Tensor *out_tensor = nullptr;
  if (!inplace) {
    out_tensor = new Tensor(tensor, tensor->require_grad());
    out_data = out_tensor->data();
  }
  switch (tensor->state()) {
    case stensor::CPU:stensor::cpu_add_scalar(out_tensor->size(), in_data, val, out_data);
      break;
    case stensor::GPU:stensor::gpu_add_scalar(out_tensor->size(), in_data, val, out_data);
      break;
  }
  return out_tensor;
}

Tensor *scale(Tensor *tensor, float val, bool inplace) {
  float *in_data = tensor->data();
  float *out_data = tensor->data();
  Tensor *out_tensor = nullptr;
  if (!inplace) {
    out_tensor = new Tensor(tensor, tensor->require_grad());
    out_data = out_tensor->data();
  }
  switch (tensor->state()) {
    case stensor::CPU:stensor::cpu_scale(out_tensor->size(), in_data, val, out_data);
      break;
    case stensor::GPU:stensor::gpu_scale(out_tensor->size(), in_data, val, out_data);
      break;
  }
  return out_tensor;
}

Tensor *pow(Tensor *tensor, float val, bool inplace) {
  float *in_data = tensor->data();
  float *out_data = tensor->data();
  Tensor *out_tensor = nullptr;
  if (!inplace) {
    out_tensor = new Tensor(tensor, tensor->require_grad());
    out_data = out_tensor->data();
  }
  switch (tensor->state()) {
    case stensor::CPU:stensor::cpu_pow_scalar(out_tensor->size(), in_data, val, out_data);
      break;
    case stensor::GPU:stensor::gpu_pow_scalar(out_tensor->size(), in_data, val, out_data);
      break;
  }
  return out_tensor;
}

Tensor *exp(Tensor *tensor, bool inplace) {
  float *in_data = tensor->data();
  float *out_data = tensor->data();
  Tensor *out_tensor = nullptr;
  if (!inplace) {
    out_tensor = new Tensor(tensor, tensor->require_grad());
    out_data = out_tensor->data();
  }
  switch (tensor->state()) {
    case stensor::CPU:stensor::cpu_exp(out_tensor->size(), in_data, out_data);
      break;
    case stensor::GPU:stensor::gpu_exp(out_tensor->size(), in_data, out_data);
      break;
  }
  return out_tensor;
}

// Tensor - Tensor

#define IMPLEMENT_BINARY_FUNC_PP(name) \
Tensor *name(const Tensor *a, const Tensor *b) {\
  CHECK_EQ(a->device(), b->device()) << "tensors must be at same device";\
  bool require_grad = (a->require_grad() || b->require_grad());\
  std::vector<int> shape_a(a->shape());\
  std::vector<int> shape_b(b->shape());\
  std::vector<int> shape_out;\
  float *out_data = nullptr;\
  Tensor *out_tensor = nullptr;\
  bool broadcast = false;\
  if (a->ShapeEquals(b))\
    shape_out = shape_a;\
  else {\
    broadcast = true;\
    shape_out = stensor::broadcast(shape_a, shape_b);\
  } \
  out_tensor = new Tensor(shape_out, a->device(), require_grad);\
  out_data = out_tensor->data();\
  switch (a->state()) {\
    case stensor::CPU:\
      if (!broadcast)\
        stensor::cpu_##name(a->size(), a->const_data(), b->const_data(),\
                         out_data);\
      else\
        stensor::cpu_##name##_broadcast(a->const_data(), b->const_data(),\
                                   shape_a, shape_b,\
                                   out_data);\
      break;\
    case stensor::GPU:\
      if (!broadcast)\
        stensor::gpu_##name(a->size(), a->const_data(), b->const_data(),\
                         out_data);\
      else\
        stensor::gpu_##name##_broadcast(a->const_data(), b->const_data(),\
                                   shape_a, shape_b,\
                                   out_data);\
      break;\
  }\
  return out_tensor;\
}

//IMPLEMENT_BINARY_FUNC_PP(add);
IMPLEMENT_BINARY_FUNC_PP(sub);
IMPLEMENT_BINARY_FUNC_PP(mul);
IMPLEMENT_BINARY_FUNC_PP(div);
IMPLEMENT_BINARY_FUNC_PP(pow);

Tensor *add(const Tensor *a, const Tensor *b) {
  CHECK_EQ(a->device(), b->device()) << "tensors must be at same device";
  bool require_grad = (a->require_grad() || b->require_grad());
  std::vector<int> shape_a(a->shape());
  std::vector<int> shape_b(b->shape());
  std::vector<int> shape_out;
  float *out_data = nullptr;
  Tensor *out_tensor = nullptr;
  bool broadcast = false;
  if (a->ShapeEquals(b))
    shape_out = shape_a;
  else {
    broadcast = true;
    shape_out = stensor::broadcast(shape_a, shape_b);
  }
  out_tensor = new Tensor(shape_out, a->device(), require_grad);
  out_data = out_tensor->data();
  switch (a->state()) {
    case stensor::CPU:
      if (!broadcast)
        stensor::cpu_add(a->size(), a->const_data(), b->const_data(),
                         out_data);
      else
        stensor::cpu_add_broadcast(a->const_data(), b->const_data(),
                                   shape_a, shape_b,
                                   out_data);
      break;
    case stensor::GPU:
      if (!broadcast)
        stensor::gpu_add(a->size(), a->const_data(), b->const_data(),
                         out_data);
      else
        stensor::gpu_add_broadcast(a->const_data(), b->const_data(),
                                   shape_a, shape_b,
                                   out_data);
      break;
  }
  return out_tensor;
}

Tensor *matmul(const Tensor *a, const Tensor *b, int axis, bool transA, bool transB) {
  CHECK_EQ(a->device(), b->device()) << "tensors must be at same device";
  // inference shape
  int start_axis_a = a->CanonicalAxisIndex(axis);
  int start_axis_b = b->CanonicalAxisIndex(axis);
  int Ma = a->count(0, start_axis_a);
  int Na = a->count(start_axis_a, a->num_axes());
  if (transA) swap(Ma, Na);

  int Mb = b->count(0, start_axis_b);
  int Nb = b->count(start_axis_b, b->num_axes());
  if (transB) swap(Mb, Nb);
  bool require_grad = (a->require_grad() || b->require_grad());

  std::vector<int> out_shape;
  for (int i = 0; i < start_axis_a; ++i)
    out_shape.push_back(a->shape(i));
  for (int i = start_axis_b; i < b->num_axes(); ++i)
    out_shape.push_back(b->shape(i));

  CHECK_EQ(Na, Mb) << "Shape mismatch";
  const CBLAS_TRANSPOSE TranA = transA ? CblasTrans : CblasNoTrans;
  const CBLAS_TRANSPOSE TranB = transB ? CblasTrans : CblasNoTrans;
  Tensor *out_tensor = new Tensor(out_shape, a->device(), require_grad);

  switch (a->state()) {
    case stensor::CPU:
      stensor::cpu_gemm(TranA, TranB, Ma, Nb, Mb,
                        1.0f, a->const_data(), b->const_data(),
                        0.0f, out_tensor->data());

      break;
    case stensor::GPU:
      stensor::gpu_gemm(TranA, TranB, Ma, Nb, Mb,
                        1.0f, a->const_data(), b->const_data(),
                        0.0f, out_tensor->data());
      break;
  }
  return out_tensor;
}

Tensor *maximum(const Tensor *a, const Tensor *b) {
  CHECK_EQ(a->device(), b->device()) << "tensors must be at same device";
  CHECK(a->ShapeEquals(b)) << "tensors must be at same shape";
  bool require_grad = (a->require_grad() || b->require_grad());
  Tensor *out_tensor = new Tensor(a->shape(), a->device(), require_grad);
  float * out_data = out_tensor->data();
  switch (a->state()) {
    case stensor::CPU:
      stensor::cpu_maximum(a->size(), a->const_data(), b->const_data(),
                           out_data);
      break;
    case stensor::GPU:
      stensor::gpu_maximum(a->size(), a->const_data(), b->const_data(),
                           out_data);
      break;
  }
  return out_tensor;
}
Tensor *minimum(const Tensor *a, const Tensor *b) {
  CHECK_EQ(a->device(), b->device()) << "tensors must be at same device";
  CHECK(a->ShapeEquals(b)) << "tensors must be at same shape";
  bool require_grad = (a->require_grad() || b->require_grad());
  Tensor *out_tensor = new Tensor(a->shape(), a->device(), require_grad);
  float * out_data = out_tensor->data();
  switch (a->state()) {
    case stensor::CPU:
      stensor::cpu_minimum(a->size(), a->const_data(), b->const_data(),
                           out_data);
      break;
    case stensor::GPU:
      stensor::gpu_minimum(a->size(), a->const_data(), b->const_data(),
                           out_data);
      break;
  }
  return out_tensor;
}


/* math of Tensor end */

/* Tensor Generator*/

Tensor *random(const Tensor::ShapeType &shape, float a, float b, int device_id, bool require_grad) {
  Tensor *new_t = new Tensor(shape, device_id, require_grad);
  switch (new_t->state()) {
    case CPU:
      cpu_rng_uniform<Tensor::Dtype>(new_t->size(),
                                     Tensor::Dtype(a),
                                     Tensor::Dtype(b),
                                     new_t->data());
      break;
    case GPU:
      gpu_rng_uniform<Tensor::Dtype>(new_t->size(),
                                     Tensor::Dtype(a),
                                     Tensor::Dtype(b),
                                     new_t->data());
      break;
    default:
      cpu_rng_uniform<Tensor::Dtype>(new_t->size(),
                                     Tensor::Dtype(a),
                                     Tensor::Dtype(b),
                                     new_t->data());
      break;
  }

  return new_t;
}

Tensor *random_gaussian(const Tensor::ShapeType &shape, float mu, float sigma, int device_id, bool require_grad) {
  Tensor *new_t = new Tensor(shape, device_id, require_grad);
  switch (new_t->state()) {
    case CPU:
      cpu_rng_gaussian<Tensor::Dtype>(new_t->size(),
                                      Tensor::Dtype(mu),
                                      Tensor::Dtype(sigma),
                                      new_t->data());
      break;
    case GPU:
      gpu_rng_gaussian<Tensor::Dtype>(new_t->size(),
                                      Tensor::Dtype(mu),
                                      Tensor::Dtype(sigma),
                                      new_t->data());
      break;
    default:
      cpu_rng_gaussian<Tensor::Dtype>(new_t->size(),
                                      Tensor::Dtype(mu),
                                      Tensor::Dtype(sigma),
                                      new_t->data());
      break;
  }
  return new_t;
}

}