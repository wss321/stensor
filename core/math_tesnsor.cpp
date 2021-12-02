/**
* Copyright 2021 wss
* Created by wss on 11月,16, 2021
*/
#include "math_tesnsor.hpp"
#include "math/math_base_cpu.hpp"
#include "math/math_base_cuda.hpp"

namespace stensor {
/* math of Tensor */
/* self-op start*/
#define CHECK_SHAPE(expect, got)\
CHECK(expect == got)\
<< "Expected " << expect\
<< ", but got " << got\

template<typename Dtype>
Tensor<Dtype> *sigmoid(const Tensor<Dtype> *in, Tensor<Dtype> *out, bool grad_op) {
  if (out == nullptr)
    out = new Tensor<Dtype> *(in->shape(),in->device(), in->require_grad());

  else {
    CHECK_EQ(in->device(), out->device()) << "tensors must be at same device";
    CHECK_SHAPE(in->shape(), out->shape());
  }
  const Dtype *in_data;
  Dtype *out_data;
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
template Tensor<float> *sigmoid<float>(const Tensor<float> *tensor, Tensor<float> *out, bool grad_op);
template Tensor<double> *sigmoid<double>(const Tensor<double> *tensor, Tensor<double> *out, bool grad_op);

#define IMPLEMENT_TENSOR_UNARY_FUNC(name) \
template<typename Dtype>                  \
Tensor<Dtype> *name(const Tensor<Dtype> *in, Tensor<Dtype> *out, bool grad_op) {\
  if (out == nullptr)\
    out = new  Tensor<Dtype> *(in->shape(),in->device(), in->require_grad());\
  else{\
    CHECK_EQ(in->device(), out->device())<< "tensors must be at same device";\
    CHECK_SHAPE(in->shape(), out->shape());\
  }\
  const Dtype *in_data;\
  Dtype *out_data;\
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

IMPLEMENT_TENSOR_UNARY_FUNC(abs);
IMPLEMENT_TENSOR_UNARY_FUNC(log);
IMPLEMENT_TENSOR_UNARY_FUNC(tanh);
IMPLEMENT_TENSOR_UNARY_FUNC(relu);
IMPLEMENT_TENSOR_UNARY_FUNC(elu);
IMPLEMENT_TENSOR_UNARY_FUNC(gelu);
IMPLEMENT_TENSOR_UNARY_FUNC(leakyrelu);
IMPLEMENT_TENSOR_UNARY_FUNC(sign);
IMPLEMENT_TENSOR_UNARY_FUNC(sqrt);
IMPLEMENT_TENSOR_UNARY_FUNC(square);
IMPLEMENT_TENSOR_UNARY_FUNC(exp);

template<typename Dtype>
Tensor<Dtype> *clamp(const Tensor<Dtype> *in, float minVal, float maxVal, Tensor<Dtype> *out, bool grad_op) {
  if (out == nullptr)
    out = new  Tensor<Dtype> *(in->shape(),in->device(), in->require_grad());
  else {
    CHECK_EQ(in->device(), out->device()) << "tensors must be at same device";
    CHECK_SHAPE(in->shape(), out->shape());
  }
  const Dtype *in_data;
  Dtype *out_data;
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

template<typename Dtype>
Tensor<Dtype> *repeat(const Tensor<Dtype> *in, int axis, int num, Tensor<Dtype> *out, bool grad_op) {
  axis = in->canonical_axis_index(axis);
  CHECK_EQ(in->shape(axis), 1) << "The shape of the repeat axis must be 1";
  CHECK_GT(num, 0);
  std::vector<int> new_shape(in->shape());
  new_shape[axis] = num;
  int num_axis = in->num_axes();
  int count_row = in->count(0, axis);
  int count_col = in->count(axis, num_axis);

  if (out == nullptr)
    out = new  Tensor<Dtype> *(new_shape, in->device(), in->require_grad());
  else {
    CHECK_EQ(in->device(), out->device()) << "tensors must be at same device";
    CHECK_SHAPE(new_shape, out->shape());
  }
  const Dtype *in_data;
  Dtype *out_data;
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
        const Dtype *head_in = &in_data[i * count_col];
        Dtype *head_out = &out_data[i * count_col * num];
        for (int j = 0; j < num; ++j) {
          cpu_copy(count_col, head_in, head_out);
          head_out += count_col;
        }
      }
      break;
    case GPU:
      for (int i = 0; i < count_row; ++i) {
        const Dtype *head_in = &in_data[i * count_col];
        Dtype *head_out = &out_data[i * count_col * num];
        for (int j = 0; j < num; ++j) {
          gpu_copy(count_col, head_in, head_out);
          head_out += count_col;
        }
      }
      break;
  }

  return out;
}

template<typename Dtype>
Tensor<Dtype> *softmax(const Tensor<Dtype> *in, int axis, Tensor<Dtype> *out, bool grad_op) {
  int caxis = in->canonical_axis_index(axis);
  if (out != nullptr)
    CHECK_EQ(in->device(), out->device()) << "tensors must be at same device";
  else
    out = new  Tensor<Dtype> *(in->shape(), in->device(), in->require_grad());

  int M = in->count(0, caxis);
  int N = in->count(caxis + 1, in->num_axes());
  int D = in->shape(caxis);
  CHECK_EQ(in->size(), out->size());
  Dtype *out_data = nullptr;
  const Dtype *in_data = nullptr;
  if (!grad_op) {
    out_data = out->data();
    in_data = in->const_data();
  } else {
    out_data = out->grad();
    in_data = in->const_grad();
  }
  switch (in->state()) {
    case CPU:cpu_softmax(M, D, N, in_data, 0.0f, out_data);
      break;
    case GPU:gpu_softmax(M, D, N, in_data, 0.0f, out_data);
      break;
  }
  return out;
}

template<typename Dtype>
Tensor<Dtype> *argmax(const Tensor<Dtype> *in, int axis, Tensor<Dtype> *out, bool grad_op) {
  int caxis = in->canonical_axis_index(axis);
  std::vector<int> new_shape(in->shape());
  new_shape[caxis] = 1;
  if (out != nullptr) {
    CHECK_EQ(in->device(), out->device()) << "tensors must be at same device";
    CHECK_SHAPE(new_shape, out->shape());
  } else
    out = new  Tensor<Dtype> *(new_shape, in->device(), in->require_grad());
  int M = in->count(0, caxis);
  int N = in->count(caxis + 1, in->num_axes());
  int D = in->shape(caxis);
  Dtype *out_data = nullptr;
  const Dtype *in_data = nullptr;
  if (!grad_op) {
    out_data = out->data();
    in_data = in->const_data();
  } else {
    out_data = out->grad();
    in_data = in->const_grad();
  }
  switch (in->state()) {
    case CPU:cpu_argmax(M, D, N, in_data, out_data);
      break;
    case GPU:gpu_argmax(M, D, N, in_data, out_data);
      break;
  }
  return out;
}

template<typename Dtype>
Tensor<Dtype> *one_hot(const Tensor<Dtype> *in, int num_class, Tensor<Dtype> *out, bool grad_op) {
  std::vector<int> out_shape(in->shape());
  out_shape.push_back(num_class);
  if (out != nullptr) {
    CHECK_EQ(in->device(), out->device()) << "tensors must be at same device";
    CHECK_SHAPE(out_shape, out->shape());
  } else
    out = new  Tensor<Dtype> *(out_shape, in->device(), in->require_grad());
  int M = in->size();
  Dtype *out_data = nullptr;
  const Dtype *in_data = nullptr;
  if (!grad_op) {
    out_data = out->data();
    in_data = in->const_data();
  } else {
    out_data = out->grad();
    in_data = in->const_grad();
  }
  switch (in->state()) {
    case CPU:cpu_one_hot(M, num_class, in_data, out_data);
      break;
    case GPU:gpu_one_hot(M, num_class, in_data, out_data);
      break;
  }
  return out;
}

/* self-op end*/
template<typename Dtype>
void set(Tensor<Dtype> *in, float val, bool grad_op) {
  Dtype * data;
  if (!grad_op)data = in->data();
  else data = in->grad();
  switch (in->state()) {
    case stensor::CPU:stensor::cpu_set(in->size(), val, data);
      break;
    case stensor::GPU:stensor::gpu_set(in->size(), val, data);
      break;
  }
}

template<typename Dtype>
Tensor<Dtype> *add(const Tensor<Dtype> *in, float val, Tensor<Dtype> *out, bool grad_op) {
  if (out == nullptr)
    out = new Tensor<Dtype> *(in->shape(),in->device(), in->require_grad());
  else {
    CHECK_EQ(in->device(), out->device()) << "tensors must be at same device";
    CHECK_SHAPE(in->shape(), out->shape());
  }
  const Dtype *in_data;
  Dtype *out_data;
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
//template<> Tensor<float> *add<float>(const Tensor<float> *in, float val, Tensor<float> *out, bool grad_op);
//template<> Tensor<double> *add<double>(const Tensor<double> *in, float val, Tensor<double> *out, bool grad_op);

#define IMPLEMENT_TENSOR_SCALAR_FUNC(name, base_math_func) \
template<typename Dtype>                                   \
Tensor<Dtype> *name(const Tensor<Dtype> *in, float val, Tensor<Dtype> *out, bool grad_op) {\
  if (out == nullptr)\
    out = new  Tensor<Dtype> *(in->shape(),in->device(), in->require_grad());\
  else{\
    CHECK_EQ(in->device(), out->device()) << "tensors must be at same device";\
    CHECK_SHAPE(in->shape(), out->shape());\
  }\
  const Dtype *in_data;\
  Dtype *out_data;\
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
template<typename Dtype>               \
Tensor<Dtype> *name(const Tensor<Dtype> *a, const Tensor<Dtype> *b, Tensor<Dtype> *out, bool grad_op){\
  CHECK_EQ(a->device(), b->device()) << "tensors must be at same device";\
  bool require_grad = (a->require_grad() || b->require_grad());\
  std::vector<int> shape_a(a->shape());\
  std::vector<int> shape_b(b->shape());\
  std::vector<int> shape_out;\
\
  Dtype *out_data = nullptr;\
  const Dtype *in_data_a = nullptr;\
  const Dtype *in_data_b = nullptr;\
  bool broadcast = false;\
  if (a->shape_equal(b))\
    shape_out = shape_a;\
  else {\
    broadcast = true;\
    shape_out = stensor::broadcast(shape_a, shape_b);\
  }\
  if (out== nullptr)\
    out = new  Tensor<Dtype> *(shape_out, a->device(), require_grad);\
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

template<typename Dtype>
Tensor<Dtype> *add(const Tensor<Dtype> *a, const Tensor<Dtype> *b, Tensor<Dtype> *out, bool grad_op) {
  CHECK_EQ(a->device(), b->device()) << "tensors must be at same device";
  bool require_grad = (a->require_grad() || b->require_grad());
  std::vector<int> shape_a(a->shape());
  std::vector<int> shape_b(b->shape());
  std::vector<int> shape_out;

  Dtype *out_data = nullptr;
  const Dtype *in_data_a = nullptr;
  const Dtype *in_data_b = nullptr;
  bool broadcast = false;
  if (a->shape_equal(b))
    shape_out = shape_a;
  else {
    broadcast = true;
    shape_out = stensor::broadcast(shape_a, shape_b);
  }
  if (out == nullptr)
    out = new  Tensor<Dtype> *(shape_out, a->device(), require_grad);
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

template<typename Dtype>
Tensor<Dtype> *matmul(const Tensor<Dtype> *a, const Tensor<Dtype> *b, int axis, bool transA, bool transB, float beta,
               Tensor<Dtype> *out, bool grad_op) {
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
    out = new  Tensor<Dtype> *(out_shape, a->device(), require_grad);
  else {
    CHECK_EQ(a->device(), out->device()) << "tensors must be at same device";
    CHECK_SHAPE(out_shape, out->shape());
  }
  Dtype *out_data = nullptr;
  const Dtype *in_data_a = nullptr;
  const Dtype *in_data_b = nullptr;
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

template<typename Dtype>
Tensor<Dtype> *maximum(const Tensor<Dtype> *a, const Tensor<Dtype> *b, Tensor<Dtype> *out, bool grad_op) {
  CHECK_EQ(a->device(), b->device()) << "tensors must be at same device";
  CHECK(a->shape_equal(b)) << "tensors must be at same shape";
  bool require_grad = (a->require_grad() || b->require_grad());
  if (out == nullptr)
    out = new  Tensor<Dtype> *(a->shape(), a->device(), require_grad);
  else {
    CHECK_EQ(a->device(), out->device()) << "tensors must be at same device";
    CHECK_SHAPE(a->shape(), out->shape());
  }
  Dtype *out_data = nullptr;
  const Dtype *in_data_a = nullptr;
  const Dtype *in_data_b = nullptr;
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

template<typename Dtype>
Tensor<Dtype> *minimum(const Tensor<Dtype> *a, const Tensor<Dtype> *b, Tensor<Dtype> *out, bool grad_op) {
  CHECK_EQ(a->device(), b->device()) << "tensors must be at same device";
  CHECK(a->shape_equal(b)) << "tensors must be at same shape";
  bool require_grad = (a->require_grad() || b->require_grad());
  if (out == nullptr)
    out = new  Tensor<Dtype> *(a->shape(), a->device(), require_grad);
  else {
    CHECK_EQ(a->device(), out->device()) << "tensors must be at same device";
    CHECK_SHAPE(a->shape(), out->shape());
  }
  Dtype *out_data = nullptr;
  const Dtype *in_data_a = nullptr;
  const Dtype *in_data_b = nullptr;
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

template<typename Dtype>
Tensor<Dtype> *concat(const std::vector<Tensor<Dtype> *> &inputs, int axis, Tensor<Dtype> *out) {
  CHECK_GE(inputs.size(), 1);
  int caxis = inputs[0]->canonical_axis_index(axis);
  std::vector<int> shape_ref(inputs[0]->shape());
  shape_ref[caxis] = 1;
  int new_dim = 0;
  bool require_grad = false;
  std::vector<float *> data_heads(inputs.size());
  std::vector<float *> grad_heads(inputs.size());

  for (int i = 0; i < inputs.size(); ++i) {
    std::vector<int> shape_cur(inputs[i]->shape());
    shape_cur[caxis] = 1;
    CHECK_SHAPE(shape_ref, shape_cur);
    CHECK_EQ(inputs[0]->device(), inputs[i]->device()) << "tensors must be at same device";
    require_grad = inputs[i]->require_grad() || require_grad;
    new_dim += inputs[i]->shape(caxis);
    data_heads[i] = inputs[i]->data();
    if (inputs[i]->require_grad())
      grad_heads[i] = inputs[i]->grad();
    else grad_heads[i] = nullptr;
  }
  shape_ref[caxis] = new_dim;
  if (out == nullptr)
    out = new  Tensor<Dtype> *(shape_ref, inputs[0]->device(), require_grad);
  else {
    CHECK_EQ(inputs[0]->device(), out->device()) << "tensors must be at same device";
    CHECK_SHAPE(shape_ref, out->shape());
  }
  int M = inputs[0]->count(0, caxis);

  Dtype *out_data = out->data();
  Dtype *out_grad = nullptr;
  if (out->require_grad())
    out_grad = out->grad();
  switch (inputs[0]->state()) {
    case stensor::CPU:
      for (int m = 0; m < M; ++m) {
        for (int l = 0; l < inputs.size(); ++l) {
          int N = inputs[l]->count(caxis, inputs[l]->num_axes());
          stensor::cpu_copy(N, data_heads[l], out_data);
          out_data += N;
          data_heads[l] += N;
          // copy grad
          if (out_grad != nullptr && grad_heads[l] != nullptr) {
            stensor::cpu_copy(N, grad_heads[l], out_grad);
            out_grad += N;
            grad_heads[l] += N;
          }
        }

      }
      break;
    case stensor::GPU:
      for (int m = 0; m < M; ++m) {
        for (int l = 0; l < inputs.size(); ++l) {
          int N = inputs[l]->count(caxis, inputs[l]->num_axes());
          stensor::gpu_copy(N, data_heads[l], out_data);
          out_data += N;
          data_heads[l] += N;
          // copy grad
          if (out_grad != nullptr && grad_heads[l] != nullptr) {
            stensor::gpu_copy(N, grad_heads[l], out_grad);
            out_grad += N;
            grad_heads[l] += N;
          }
        }

      }
      break;
  }
  return out;
}

/* math of Tensor end */

/* Tensor Generator*/
template<typename Dtype>
Tensor<Dtype> *random(const std::vector<int> &shape, float a, float b, int device_id, bool require_grad) {
  Tensor<Dtype> *new_t = new  Tensor<Dtype> *(shape, device_id, require_grad);
  switch (new_t->state()) {
    case CPU:
      cpu_rng_uniform<Tensor<Dtype>::Dtype>(new_t->size(),
                                     Tensor<Dtype>::Dtype(a),
                                     Tensor<Dtype>::Dtype(b),
                                     new_t->data());
      break;
    case GPU:
      gpu_rng_uniform<Tensor<Dtype>::Dtype>(new_t->size(),
                                     Tensor<Dtype>::Dtype(a),
                                     Tensor<Dtype>::Dtype(b),
                                     new_t->data());
      break;
    default:
      cpu_rng_uniform<Tensor<Dtype>::Dtype>(new_t->size(),
                                     Tensor<Dtype>::Dtype(a),
                                     Tensor<Dtype>::Dtype(b),
                                     new_t->data());
      break;
  }

  return new_t;
}

template<typename Dtype>
Tensor<Dtype> *random_gaussian(const std::vector<int> &shape, float mu, float sigma, int device_id, bool require_grad) {
  Tensor<Dtype> *new_t = new  Tensor<Dtype> *(shape, device_id, require_grad);
  switch (new_t->state()) {
    case CPU:
      cpu_rng_gaussian<Tensor<Dtype>::Dtype>(new_t->size(),
                                      Tensor<Dtype>::Dtype(mu),
                                      Tensor<Dtype>::Dtype(sigma),
                                      new_t->data());
      break;
    case GPU:
      gpu_rng_gaussian<Tensor<Dtype>::Dtype>(new_t->size(),
                                      Tensor<Dtype>::Dtype(mu),
                                      Tensor<Dtype>::Dtype(sigma),
                                      new_t->data());
      break;
    default:
      cpu_rng_gaussian<Tensor<Dtype>::Dtype>(new_t->size(),
                                      Tensor<Dtype>::Dtype(mu),
                                      Tensor<Dtype>::Dtype(sigma),
                                      new_t->data());
      break;
  }
  return new_t;
}

// reduction sum of axis
template<typename Dtype>
Tensor<Dtype> *sum(const Tensor<Dtype> *a, int axis, Tensor<Dtype> *out, bool grad_op) {
  int caxis = a->canonical_axis_index(axis);
  if (out != nullptr)
    CHECK_EQ(a->device(), out->device()) << "tensors must be at same device";
  else {
    std::vector<int> new_shape(a->shape());
    new_shape[caxis] = 1;
    out = new  Tensor<Dtype> *(new_shape, a->device(), a->require_grad());
  }

  int M = a->count(0, caxis);
  int N = a->count(caxis + 1, a->num_axes());
  int D = a->shape(caxis);
  CHECK_EQ(M * N, out->size());
  Dtype *out_data = nullptr;
  const Dtype *in_data = nullptr;
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
    case GPU:gpu_reduce_sum(M, D, N, in_data, 0.0f, out_data);
//      for (int i = 0; i < M; ++i) {
//        gpu_gemm(false, false, 1, N, D, 1.0f, sum_multiplier->data(), in_data, 0.0f, out_data);
//        out_data += N;
//        in_data += D * N;
//      }
      break;
  }
  return out;
}

template<typename Dtype>
Tensor<Dtype> *mean(const Tensor<Dtype> *a, int axis, Tensor<Dtype> *out, bool grad_op) {
  int caxis = a->canonical_axis_index(axis);
  if (out != nullptr)
    CHECK_EQ(a->device(), out->device()) << "tensors must be at same device";
  else {
    std::vector<int> new_shape(a->shape());
    new_shape[caxis] = 1;
    out = new  Tensor<Dtype> *(new_shape, a->device(), a->require_grad());
  }

  int M = a->count(0, caxis);
  int N = a->count(caxis + 1, a->num_axes());
  int D = a->shape(caxis);
  CHECK_EQ(M * N, out->size());
  Dtype *out_data = nullptr;
  const Dtype *in_data = nullptr;
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
    case GPU:gpu_reduce_mean(M, D, N, in_data, 0.0f, out_data);
      break;
  }
  return out;
}

template<typename Dtype>
Tensor<Dtype> *asum(const Tensor<Dtype> *a, int axis, Tensor<Dtype> *out, bool grad_op) {
  int caxis = a->canonical_axis_index(axis);
  if (out != nullptr)
    CHECK_EQ(a->device(), out->device()) << "tensors must be at same device";
  else {
    std::vector<int> new_shape(a->shape());
    new_shape[caxis] = 1;
    out = new  Tensor<Dtype> *(new_shape, a->device(), a->require_grad());
  }

  int M = a->count(0, caxis);
  int N = a->count(caxis + 1, a->num_axes());
  int D = a->shape(caxis);
  CHECK_EQ(M * N, out->size());
  Tensor<Dtype> *sum_multiplier = stensor::ones({D}, a->device(), false);
  Dtype *out_data = nullptr;
  const Dtype *in_data = nullptr;
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
    case GPU:gpu_reduce_asum(M, D, N, in_data, 0.0f, out_data);
      break;
  }
  return out;
}
}//namespace stensor