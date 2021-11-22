/**
* Copyright 2021 wss
* Created by wss on 11月,16, 2021
*/
#include "math_tesnsor.hpp"

namespace stensor {
/* math of Tensor */
/* self-op start*/
#define CHECK_SHAPE(expect, got)\
CHECK(expect == got)\
<< "Expected " << expect\
<< ", but got " << got\

Tensor *sigmoid(const Tensor *in, Tensor *out, bool grad_op) {
  if (out == nullptr)
    out = new Tensor(in, in->require_grad());
  else {
    CHECK_EQ(in->device(), out->device()) << "tensors must be at same device";
    CHECK_SHAPE(in->shape(), out->shape());
  }
  const float *in_data;
  float *out_data;
  if (!grad_op) {
    in_data = in->const_data();
    out_data = out->data();
  } else {
    in_data = in->const_grad();
    out_data = out->grad();
  }
  switch (in->state()) {
    case stensor::CPU:stensor::cpu_sigmoid(in->size(), in_data, out_data);
      break;
    case stensor::GPU:stensor::gpu_sigmoid(in->size(), in_data, out_data);
      break;
  }
  return out;
}
#define IMPLEMENT_TENSOR_UNARY_FUNC(name) \
Tensor *name(const Tensor *in, Tensor *out, bool grad_op) {\
  if (out == nullptr)\
    out = new Tensor(in, in->require_grad());\
  else{\
    CHECK_EQ(in->device(), out->device())<< "tensors must be at same device";\
    CHECK_SHAPE(in->shape(), out->shape());\
  }\
  const float *in_data;\
  float *out_data;\
  if (!grad_op) {\
    in_data = in->const_data();\
    out_data = out->data();\
  } else {\
    in_data = in->const_grad();\
    out_data = out->grad();\
  }\
  switch (in->state()) {\
    case stensor::CPU:stensor::cpu_##name(in->size(), in_data, out_data);\
      break;\
    case stensor::GPU:stensor::gpu_##name(in->size(), in_data, out_data);\
      break;\
  }\
  return out;\
}

IMPLEMENT_TENSOR_UNARY_FUNC(tanh);
IMPLEMENT_TENSOR_UNARY_FUNC(relu);
IMPLEMENT_TENSOR_UNARY_FUNC(elu);
IMPLEMENT_TENSOR_UNARY_FUNC(gelu);
IMPLEMENT_TENSOR_UNARY_FUNC(leakyrelu);
IMPLEMENT_TENSOR_UNARY_FUNC(sign);
IMPLEMENT_TENSOR_UNARY_FUNC(sqrt);
IMPLEMENT_TENSOR_UNARY_FUNC(square);
IMPLEMENT_TENSOR_UNARY_FUNC(exp);

Tensor *clamp(const Tensor *in, float minVal, float maxVal, Tensor *out, bool grad_op) {
  if (out == nullptr)
    out = new Tensor(in, in->require_grad());
  else {
    CHECK_EQ(in->device(), out->device()) << "tensors must be at same device";
    CHECK_SHAPE(in->shape(), out->shape());
  }
  const float *in_data;
  float *out_data;
  if (!grad_op) {
    in_data = in->const_data();
    out_data = out->data();
  } else {
    in_data = in->const_grad();
    out_data = out->grad();
  }
  switch (in->state()) {
    case stensor::CPU:stensor::cpu_clamp(in->size(), minVal, maxVal, in_data, out_data);
      break;
    case stensor::GPU:stensor::gpu_clamp(in->size(), minVal, maxVal, in_data, out_data);
      break;
  }
  return out;
}

Tensor *repeat(const Tensor *in, int axis, int num, Tensor *out, bool grad_op) {
  axis = in->canonical_axis_index(axis);
  CHECK_EQ(in->shape(axis), 1) << "The shape of the repeat axis must be 1";
  CHECK_GT(num, 0);
  std::vector<int> new_shape(in->shape());
  new_shape[axis] = num;
  int num_axis = in->num_axes();
  int count_row = in->count(0, axis);
  int count_col = in->count(axis, num_axis);

  if (out == nullptr)
    out = new Tensor(new_shape, in->device(), in->require_grad());
  else {
    CHECK_EQ(in->device(), out->device()) << "tensors must be at same device";
    CHECK_SHAPE(new_shape, out->shape());
  }
  const float *in_data;
  float *out_data;
  if (!grad_op) {
    in_data = in->const_data();
    out_data = out->data();
  } else {
    in_data = in->const_grad();
    out_data = out->grad();
  }

  switch (in->state()) {
    case CPU:
      for (int i = 0; i < count_row; ++i) {
        const float *head_in = &in_data[i * count_col];
        float *head_out = &out_data[i * count_col * num];
        for (int j = 0; j < num; ++j) {
          cpu_copy(count_col, head_in, head_out);
          head_out += count_col;
        }
      }
      break;
    case GPU:
      for (int i = 0; i < count_row; ++i) {
        const float *head_in = &in_data[i * count_col];
        float *head_out = &out_data[i * count_col * num];
        for (int j = 0; j < num; ++j) {
          gpu_copy(count_col, head_in, head_out);
          head_out += count_col;
        }
      }
      break;
  }

  return out;
}

/* self-op end*/

void set(Tensor *in, float val, bool grad_op) {
  Tensor::Dtype *data;
  if (!grad_op)data = in->data();
  else data = in->grad();
  switch (in->state()) {
    case stensor::CPU:stensor::cpu_set(in->size(), val, data);
      break;
    case stensor::GPU:stensor::gpu_set(in->size(), val, data);
      break;
  }
}

Tensor *add(const Tensor *in, float val, Tensor *out, bool grad_op) {
  if (out == nullptr)
    out = new Tensor(in, in->require_grad());
  else {
    CHECK_EQ(in->device(), out->device()) << "tensors must be at same device";
    CHECK_SHAPE(in->shape(), out->shape());
  }
  const float *in_data;
  float *out_data;
  if (!grad_op) {
    in_data = in->const_data();
    out_data = out->data();
  } else {
    in_data = in->const_grad();
    out_data = out->grad();
  }
  switch (in->state()) {
    case stensor::CPU:stensor::cpu_add_scalar(out->size(), in_data, val, out_data);
      break;
    case stensor::GPU:stensor::gpu_add_scalar(out->size(), in_data, val, out_data);
      break;
  }
  return out;
}

#define IMPLEMENT_TENSOR_SCALAR_FUNC(name, base_math_func)\
Tensor *name(const Tensor *in, float val, Tensor *out, bool grad_op) {\
  if (out == nullptr)\
    out = new Tensor(in, in->require_grad());\
  else{\
    CHECK_EQ(in->device(), out->device()) << "tensors must be at same device";\
    CHECK_SHAPE(in->shape(), out->shape());\
  }\
  const float *in_data;\
  float *out_data;\
  if (!grad_op) {\
    in_data = in->const_data();\
    out_data = out->data();\
  } else {\
    in_data = in->const_grad();\
    out_data = out->grad();\
  }\
  switch (in->state()) {\
    case stensor::CPU:stensor::cpu_##base_math_func(out->size(), in_data, val, out_data);\
      break;\
    case stensor::GPU:stensor::gpu_##base_math_func(out->size(), in_data, val, out_data);\
      break;\
  }\
  return out;\
}

IMPLEMENT_TENSOR_SCALAR_FUNC(scale, scale);
IMPLEMENT_TENSOR_SCALAR_FUNC(pow, pow_scalar);

// Tensor - Tensor

#define IMPLEMENT_BINARY_FUNC_PP(name) \
Tensor *name(const Tensor *a, const Tensor *b, Tensor *out, bool grad_op){\
  CHECK_EQ(a->device(), b->device()) << "tensors must be at same device";\
  bool require_grad = (a->require_grad() || b->require_grad());\
  std::vector<int> shape_a(a->shape());\
  std::vector<int> shape_b(b->shape());\
  std::vector<int> shape_out;\
\
  float *out_data = nullptr;\
  const float *in_data_a = nullptr;\
  const float *in_data_b = nullptr;\
  bool broadcast = false;\
  if (a->shape_equal(b))\
    shape_out = shape_a;\
  else {\
    broadcast = true;\
    shape_out = stensor::broadcast(shape_a, shape_b);\
  }\
  if (out== nullptr)\
    out = new Tensor(shape_out, a->device(), require_grad);\
  else {\
    CHECK_EQ(a->device(), out->device()) << "tensors must be at same device";\
    CHECK_SHAPE(shape_out, out->shape());\
  }\
  if (!grad_op) {\
    in_data_a =a->const_data();\
    in_data_b =b->const_data();\
    out_data = out->data();\
  }else {\
    in_data_a =a->const_grad();\
    in_data_b =b->const_grad();\
    out_data = out->grad();\
  }\
\
  switch (a->state()) {\
    case stensor::CPU:\
      if (!broadcast)\
        stensor::cpu_##name(a->size(), in_data_a, in_data_b,\
                         out_data);\
      else\
        stensor::cpu_##name##_broadcast(in_data_a, in_data_b,\
                                   shape_a, shape_b,\
                                   out_data);\
      break;\
    case stensor::GPU:\
      if (!broadcast)\
        stensor::gpu_##name(a->size(),in_data_a, in_data_b,\
                         out_data);\
      else\
        stensor::gpu_##name##_broadcast(in_data_a, in_data_b,\
                                   shape_a, shape_b,\
                                   out_data);\
      break;\
  }\
  return out;\
}

//IMPLEMENT_BINARY_FUNC_PP(add);
IMPLEMENT_BINARY_FUNC_PP(sub);
IMPLEMENT_BINARY_FUNC_PP(mul);
IMPLEMENT_BINARY_FUNC_PP(div);
IMPLEMENT_BINARY_FUNC_PP(pow);

Tensor *add(const Tensor *a, const Tensor *b, Tensor *out, bool grad_op) {
  CHECK_EQ(a->device(), b->device()) << "tensors must be at same device";
  bool require_grad = (a->require_grad() || b->require_grad());
  std::vector<int> shape_a(a->shape());
  std::vector<int> shape_b(b->shape());
  std::vector<int> shape_out;

  float *out_data = nullptr;
  const float *in_data_a = nullptr;
  const float *in_data_b = nullptr;
  bool broadcast = false;
  if (a->shape_equal(b))
    shape_out = shape_a;
  else {
    broadcast = true;
    shape_out = stensor::broadcast(shape_a, shape_b);
  }
  if (out == nullptr)
    out = new Tensor(shape_out, a->device(), require_grad);
  else {
    CHECK_EQ(a->device(), out->device()) << "tensors must be at same device";
    CHECK_SHAPE(shape_out, out->shape());
  }
  if (!grad_op) {
    in_data_a = a->const_data();
    in_data_b = b->const_data();
    out_data = out->data();
  } else {
    in_data_a = a->const_grad();
    in_data_b = b->const_grad();
    out_data = out->grad();
  }

  switch (a->state()) {
    case stensor::CPU:
      if (!broadcast)
        stensor::cpu_add(a->size(), in_data_a, in_data_b,
                         out_data);
      else
        stensor::cpu_add_broadcast(in_data_a, in_data_b,
                                   shape_a, shape_b,
                                   out_data);
      break;
    case stensor::GPU:
      if (!broadcast)
        stensor::gpu_add(a->size(), in_data_a, in_data_b,
                         out_data);
      else
        stensor::gpu_add_broadcast(in_data_a, in_data_b,
                                   shape_a, shape_b,
                                   out_data);
      break;
  }
  return out;
}

Tensor *matmul(const Tensor *a, const Tensor *b, int axis, bool transA, bool transB, float beta,
               Tensor *out, bool grad_op) {
  CHECK_EQ(a->device(), b->device()) << "tensors must be at same device";
  // inference shape
  int start_axis_a = a->canonical_axis_index(axis);
  int start_axis_b = b->canonical_axis_index(axis);
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
  if (out == nullptr)
    out = new Tensor(out_shape, a->device(), require_grad);
  else {
    CHECK_EQ(a->device(), out->device()) << "tensors must be at same device";
    CHECK_SHAPE(out_shape, out->shape());
  }
  float *out_data = nullptr;
  const float *in_data_a = nullptr;
  const float *in_data_b = nullptr;
  if (!grad_op) {
    in_data_a = a->const_data();
    in_data_b = b->const_data();
    out_data = out->data();
  } else {
    in_data_a = a->const_grad();
    in_data_b = b->const_grad();
    out_data = out->grad();
  }
  switch (a->state()) {
    case stensor::CPU:
      stensor::cpu_gemm(transA, transB, Ma, Nb, Mb,
                        1.0f, in_data_a, in_data_b,
                        beta, out_data);

      break;
    case stensor::GPU:
      stensor::gpu_gemm(transA, transB, Ma, Nb, Mb,
                        1.0f, in_data_a, in_data_b,
                        beta, out_data);
      break;
  }
  return out;
}

Tensor *maximum(const Tensor *a, const Tensor *b, Tensor *out, bool grad_op) {
  CHECK_EQ(a->device(), b->device()) << "tensors must be at same device";
  CHECK(a->shape_equal(b)) << "tensors must be at same shape";
  bool require_grad = (a->require_grad() || b->require_grad());
  if (out == nullptr)
    out = new Tensor(a->shape(), a->device(), require_grad);
  else {
    CHECK_EQ(a->device(), out->device()) << "tensors must be at same device";
    CHECK_SHAPE(a->shape(), out->shape());
  }
  float *out_data = nullptr;
  const float *in_data_a = nullptr;
  const float *in_data_b = nullptr;
  if (!grad_op) {
    in_data_a = a->const_data();
    in_data_b = b->const_data();
    out_data = out->data();
  } else {
    in_data_a = a->const_grad();
    in_data_b = b->const_grad();
    out_data = out->grad();
  }

  switch (a->state()) {
    case stensor::CPU:
      stensor::cpu_maximum(a->size(), in_data_a, in_data_b,
                           out_data);
      break;
    case stensor::GPU:
      stensor::gpu_maximum(a->size(), in_data_a, in_data_b,
                           out_data);
      break;
  }
  return out;
}

Tensor *minimum(const Tensor *a, const Tensor *b, Tensor *out, bool grad_op) {
  CHECK_EQ(a->device(), b->device()) << "tensors must be at same device";
  CHECK(a->shape_equal(b)) << "tensors must be at same shape";
  bool require_grad = (a->require_grad() || b->require_grad());
  if (out == nullptr)
    out = new Tensor(a->shape(), a->device(), require_grad);
  else {
    CHECK_EQ(a->device(), out->device()) << "tensors must be at same device";
    CHECK_SHAPE(a->shape(), out->shape());
  }
  float *out_data = nullptr;
  const float *in_data_a = nullptr;
  const float *in_data_b = nullptr;
  if (!grad_op) {
    in_data_a = a->const_data();
    in_data_b = b->const_data();
    out_data = out->data();
  } else {
    in_data_a = a->const_grad();
    in_data_b = b->const_grad();
    out_data = out->grad();
  }

  switch (a->state()) {
    case stensor::CPU:
      stensor::cpu_minimum(a->size(), in_data_a, in_data_b,
                           out_data);
      break;
    case stensor::GPU:
      stensor::gpu_minimum(a->size(), in_data_a, in_data_b,
                           out_data);
      break;
  }
  return out;
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

// reduction sum of axis
Tensor *sum(const Tensor *a, int axis, Tensor *out, bool grad_op) {
  int caxis = a->canonical_axis_index(axis);
  if (out != nullptr)
    CHECK_EQ(a->device(), out->device()) << "tensors must be at same device";
  else {
    std::vector<int> new_shape(a->shape());
    new_shape[caxis] = 1;
    out = new Tensor(new_shape, a->device(), a->require_grad());
  }

  int M = a->count(0, caxis);
  int N = a->count(caxis + 1, a->num_axes());
  int D = a->shape(caxis);
  CHECK_EQ(M * N, out->size());
  Tensor *sum_multiplier = stensor::ones({D}, a->device(), false);
  float *out_data = nullptr;
  const float *in_data = nullptr;
  if (!grad_op) {
    out_data = out->data();
    in_data = a->const_data();
  } else {
    out_data = out->grad();
    in_data = a->const_grad();
  }
  switch (a->state()) {
    case CPU:cpu_reduce_sum(M, D, N, in_data, 0.0f, out_data);
      break;
    case GPU:
      gpu_reduce_sum(M, D, N, in_data, 0.0f, out_data);
//      for (int i = 0; i < M; ++i) {
//        gpu_gemm(false, false, 1, N, D, 1.0f, sum_multiplier->data(), in_data, 0.0f, out_data);
//        out_data += N;
//        in_data += D * N;
//      }
      break;
  }
  return out;
}

Tensor *mean(const Tensor *a, int axis, Tensor *out, bool grad_op) {
  int caxis = a->canonical_axis_index(axis);
  if (out != nullptr)
    CHECK_EQ(a->device(), out->device()) << "tensors must be at same device";
  else {
    std::vector<int> new_shape(a->shape());
    new_shape[caxis] = 1;
    out = new Tensor(new_shape, a->device(), a->require_grad());
  }

  int M = a->count(0, caxis);
  int N = a->count(caxis + 1, a->num_axes());
  int D = a->shape(caxis);
  CHECK_EQ(M * N, out->size());
  Tensor *sum_multiplier = stensor::ones({D}, a->device(), false);
  float *out_data = nullptr;
  const float *in_data = nullptr;
  if (!grad_op) {
    out_data = out->data();
    in_data = a->const_data();
  } else {
    out_data = out->grad();
    in_data = a->const_grad();
  }
  switch (a->state()) {
    case CPU:cpu_reduce_mean(M, D, N, in_data, 0.0f, out_data);
      break;
    case GPU:
      gpu_reduce_mean(M, D, N, in_data, 0.0f, out_data);
      break;
  }
  return out;
}


Tensor *asum(const Tensor *a, int axis, Tensor *out, bool grad_op) {
  int caxis = a->canonical_axis_index(axis);
  if (out != nullptr)
    CHECK_EQ(a->device(), out->device()) << "tensors must be at same device";
  else {
    std::vector<int> new_shape(a->shape());
    new_shape[caxis] = 1;
    out = new Tensor(new_shape, a->device(), a->require_grad());
  }

  int M = a->count(0, caxis);
  int N = a->count(caxis + 1, a->num_axes());
  int D = a->shape(caxis);
  CHECK_EQ(M * N, out->size());
  Tensor *sum_multiplier = stensor::ones({D}, a->device(), false);
  float *out_data = nullptr;
  const float *in_data = nullptr;
  if (!grad_op) {
    out_data = out->data();
    in_data = a->const_data();
  } else {
    out_data = out->grad();
    in_data = a->const_grad();
  }
  switch (a->state()) {
    case CPU:cpu_reduce_asum(M, D, N, in_data, 0.0f, out_data);
      break;
    case GPU:
      gpu_reduce_asum(M, D, N, in_data, 0.0f, out_data);
      break;
  }
  return out;
}
}//namespace stensor