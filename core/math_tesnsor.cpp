/**
* Copyright 2021 wss
* Created by wss on 11月,16, 2021
*/
#include "math_tesnsor.hpp"

namespace stensor {
/* math of Tensor */
/* self-op start*/

Tensor *sigmoid(Tensor *tensor, bool inplace) {
  switch (tensor->state()) {
    case stensor::CPU:
      if (inplace) {
        float *data = tensor->mutable_cpu_data();
        stensor::cpu_sigmoid(tensor->size(), data, data);
        return tensor;
      } else {
        const float *data = tensor->cpu_data();
        Tensor *out_tensor = new Tensor(tensor, tensor->require_grad());
        Tensor::Dtype *out_data = out_tensor->mutable_cpu_data();
        stensor::cpu_sigmoid(out_tensor->size(), data, out_data);
        return out_tensor;
      }
      break;
    case stensor::GPU:
      if (inplace) {
        float *data = tensor->mutable_gpu_data();
        stensor::gpu_sigmoid(tensor->size(), data, data);
        return tensor;
      } else {
        const Tensor::Dtype *data = tensor->gpu_data();
        Tensor *out_tensor = new Tensor(tensor, tensor->require_grad());
        float *out_data = out_tensor->mutable_gpu_data();
        stensor::gpu_sigmoid(out_tensor->size(), data, out_data);
        return out_tensor;
      }
      break;
  }
}

#define IMPLEMENT_TENSOR_UNARY_FUNC(name) \
Tensor *name(Tensor *tensor, bool inplace) { \
  switch (tensor->state()) { \
    case stensor::CPU: \
      if (inplace) { \
        float *data = tensor->mutable_cpu_data(); \
        stensor::cpu_##name(tensor->size(), data, data); \
        return tensor; \
      } else { \
        const float *data = tensor->cpu_data(); \
        Tensor *out_tensor = new Tensor(tensor, tensor->require_grad()); \
        Tensor::Dtype *out_data = out_tensor->mutable_cpu_data(); \
        stensor::cpu_##name(out_tensor->size(), data, out_data); \
        return out_tensor; \
      } \
      break; \
    case stensor::GPU: \
      if (inplace) { \
        float *data = tensor->mutable_gpu_data(); \
        stensor::gpu_##name(tensor->size(), data, data); \
        return tensor; \
      } else { \
        const Tensor::Dtype *data = tensor->gpu_data(); \
        Tensor *out_tensor = new Tensor(tensor, tensor->require_grad()); \
        float *out_data = out_tensor->mutable_gpu_data(); \
        stensor::gpu_##name(out_tensor->size(), data, out_data); \
        return out_tensor; \
      } \
      break; \
  } \
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
  switch (tensor->state()) {
    case stensor::CPU:
      if (inplace) {
        float *data = tensor->mutable_cpu_data();
        stensor::cpu_clamp(tensor->size(), minVal, maxVal, data, data);
        return tensor;
      } else {
        const float *data = tensor->cpu_data();
        Tensor *out_tensor = new Tensor(tensor, tensor->require_grad());
        Tensor::Dtype *out_data = out_tensor->mutable_cpu_data();
        stensor::cpu_clamp(out_tensor->size(), minVal, maxVal, data, out_data);
        return out_tensor;
      }
      break;
    case stensor::GPU:
      if (inplace) {
        float *data = tensor->mutable_gpu_data();
        stensor::gpu_clamp(tensor->size(), minVal, maxVal, data, data);
        return tensor;
      } else {
        const Tensor::Dtype *data = tensor->gpu_data();
        Tensor *out_tensor = new Tensor(tensor, tensor->require_grad());
        float *out_data = out_tensor->mutable_gpu_data();
        stensor::gpu_clamp(out_tensor->size(), minVal, maxVal, data, out_data);
        return out_tensor;
      }
      break;
  }
}

/* self-op end*/

void set(Tensor &tensor, float val) {
  Tensor::Dtype *data;
  switch (tensor.state()) {
    case stensor::CPU:data = tensor.mutable_cpu_data();
      stensor::cpu_set(tensor.size(), val, data);
      break;
    case stensor::GPU:data = tensor.mutable_gpu_data();
      stensor::gpu_set(tensor.size(), val, data);
      break;
  }
}

void set(Tensor *tensor, float val) {
  float *data;
  switch (tensor->state()) {
    case stensor::CPU:data = tensor->mutable_cpu_data();
      stensor::cpu_set(tensor->size(), val, data);
      break;
    case stensor::GPU:data = tensor->mutable_gpu_data();
      stensor::gpu_set(tensor->size(), val, data);
      break;
  }
}

Tensor *add(Tensor *tensor, float val, bool inplace) {
  switch (tensor->state()) {
    case stensor::CPU:
      if (inplace) {
        float *data = tensor->mutable_cpu_data();
        stensor::cpu_add_scalar(tensor->size(), data, val, data);
        return tensor;
      } else {
        const float *data = tensor->cpu_data();
        Tensor *out_tensor = new Tensor(tensor, tensor->require_grad());
        Tensor::Dtype *out_data = out_tensor->mutable_cpu_data();
        stensor::cpu_add_scalar(out_tensor->size(), data, val, out_data);
        return out_tensor;
      }
      break;
    case stensor::GPU:
      if (inplace) {
        float *data = tensor->mutable_gpu_data();
        stensor::gpu_add_scalar(tensor->size(), data, val, data);
        return tensor;
      } else {
        const Tensor::Dtype *data = tensor->gpu_data();
        Tensor *out_tensor = new Tensor(tensor, tensor->require_grad());
        float *out_data = out_tensor->mutable_gpu_data();
        stensor::gpu_add_scalar(out_tensor->size(), data, val, out_data);
        return out_tensor;
      }
      break;
  }
}

Tensor *scale(Tensor *tensor, float val, bool inplace) {
  switch (tensor->state()) {
    case stensor::CPU:
      if (inplace) {
        float *data = tensor->mutable_cpu_data();
        stensor::cpu_scale(tensor->size(), data, val, data);
        return tensor;
      } else {
        const float *data = tensor->cpu_data();
        Tensor *out_tensor = new Tensor(tensor, tensor->require_grad());
        float *out_data = out_tensor->mutable_cpu_data();
        stensor::cpu_scale(out_tensor->size(), data, val, out_data);
        return out_tensor;
      }
    case stensor::GPU:
      if (inplace) {
        float *data = tensor->mutable_gpu_data();
        stensor::gpu_scale(tensor->size(), data, val, data);
        return tensor;
      } else {
        const float *data = tensor->cpu_data();
        Tensor *out_tensor = new Tensor(tensor, tensor->require_grad());
        float *out_data = out_tensor->mutable_gpu_data();
        stensor::gpu_scale(out_tensor->size(), data, val, out_data);
        return out_tensor;
      }
      break;
  }
}

Tensor *pow(Tensor *tensor, float val, bool inplace) {
  switch (tensor->state()) {
    case stensor::CPU:
      if (inplace) {
        float *data = tensor->mutable_cpu_data();
        stensor::cpu_pow_scalar(tensor->size(), data, val, data);
        return tensor;
      } else {
        const float *data = tensor->cpu_data();
        Tensor *out_tensor = new Tensor(tensor, tensor->require_grad());
        float *out_data = out_tensor->mutable_cpu_data();
        stensor::cpu_pow_scalar(out_tensor->size(), data, val, out_data);
        return out_tensor;
      }
      break;
    case stensor::GPU:
      if (inplace) {
        float *data = tensor->mutable_gpu_data();
        stensor::gpu_pow_scalar(tensor->size(), data, val, data);
        return tensor;
      } else {
        const float *data = tensor->gpu_data();
        Tensor *out_tensor = new Tensor(tensor, tensor->require_grad());
        float *out_data = out_tensor->mutable_gpu_data();
        stensor::gpu_pow_scalar(out_tensor->size(), data, val, out_data);
        return out_tensor;
      }
      break;
  }
}

Tensor *exp(Tensor *tensor, bool inplace) {
  switch (tensor->state()) {
    case stensor::CPU:
      if (inplace) {
        float *data = tensor->mutable_cpu_data();
        stensor::cpu_exp(tensor->size(), data, data);
        return tensor;
      } else {
        const float *data = tensor->cpu_data();
        Tensor *out_tensor = new Tensor(tensor, tensor->require_grad());
        float *out_data = out_tensor->mutable_cpu_data();
        stensor::cpu_exp(out_tensor->size(), data, out_data);
        return out_tensor;
      }
      break;
    case stensor::GPU:
      if (inplace) {
        float *data = tensor->mutable_gpu_data();
        stensor::gpu_exp(tensor->size(), data, data);
        return tensor;
      } else {
        const float *data = tensor->gpu_data();
        Tensor *out_tensor = new Tensor(tensor, tensor->require_grad());
        float *out_data = out_tensor->mutable_gpu_data();
        stensor::gpu_exp(out_tensor->size(), data, out_data);
        return out_tensor;
      }
  }
}

// Tensor - Tensor

inline void VecUint32ToVecInt(const std::vector<uint32_t> &shape_u, std::vector<int> &shape_i) {
  shape_i.resize(shape_u.size());
  for (int i = 0; i < shape_u.size(); ++i)
    shape_i[i] = static_cast<int>(shape_u[i]);
}
inline void VecIntToUint32(const std::vector<int> &shape_i, std::vector<uint32_t> &shape_u) {
  shape_u.resize(shape_i.size());
  for (int i = 0; i < shape_u.size(); ++i) {
    CHECK_GE(shape_i[i], 0);
    shape_u[i] = static_cast<uint32_t>(shape_i[i]);
  }
}

#define IMPLEMENT_BINARY_FUNC_PP(name) \
Tensor* name(const Tensor *a, const Tensor *b) { \
  CHECK_EQ(a->state(), b->state()) << "tensors must be at same device";\
  switch (a->state()) {\
    case stensor::CPU:\
      if (a->ShapeEquals(b)) {\
        bool require_grad = (a->require_grad() || b->require_grad());\
        Tensor *out_tensor = new Tensor(a->shape(), -1, require_grad);\
        stensor::cpu_##name(a->size(), a->cpu_data(), b->cpu_data(),   \
        out_tensor->mutable_cpu_data());\
        return out_tensor;\
      } else { \
        std::vector<int> shape_a;          \
        VecUint32ToVecInt(a->shape(), shape_a);\
        std::vector<int> shape_b;          \
        VecUint32ToVecInt(b->shape(), shape_b);  \
        const std::vector<int> shape_out = stensor::broadcast(shape_a, shape_b);\
        std::vector<uint32_t> shape_out_u;        \
        VecIntToUint32(shape_out, shape_out_u);  \
        bool require_grad = (a->require_grad() || b->require_grad());\
        Tensor *out_tensor = new Tensor(shape_out_u, -1, require_grad);\
        stensor::cpu_##name##_broadcast(a->cpu_data(), b->cpu_data(), shape_a, shape_b,              \
        out_tensor->mutable_cpu_data());\
        return out_tensor;\
      }\
      break;\
    case stensor::GPU:\
      CHECK_EQ(a->device(), b->device()) << "tensors must be at same device";   \
      if (a->ShapeEquals(b)) {\
        bool require_grad = (a->require_grad() || b->require_grad());\
        Tensor *out_tensor = new Tensor(a->shape(), a->device(), require_grad);\
        stensor::gpu_##name(a->size(), a->gpu_data(), b->gpu_data(),   \
        out_tensor->mutable_gpu_data());\
        return out_tensor;\
      } else {\
        std::vector<int> shape_a;          \
        VecUint32ToVecInt(a->shape(), shape_a);\
        std::vector<int> shape_b;          \
        VecUint32ToVecInt(b->shape(), shape_b);  \
        const std::vector<int> shape_out = stensor::broadcast(shape_a, shape_b);\
        std::vector<uint32_t> shape_out_u;        \
        VecIntToUint32(shape_out, shape_out_u);  \
        bool require_grad = (a->require_grad() || b->require_grad());\
        Tensor *out_tensor = new Tensor(shape_out_u, a->device(), require_grad);\
        gpu_##name##_broadcast<float>(a->gpu_data(), b->gpu_data(),    \
        shape_a, shape_b, out_tensor->mutable_gpu_data());\
        return out_tensor;\
      }\
      break;\
  }\
}

//IMPLEMENT_BINARY_FUNC_PP(add);
IMPLEMENT_BINARY_FUNC_PP(sub);
IMPLEMENT_BINARY_FUNC_PP(mul);
IMPLEMENT_BINARY_FUNC_PP(div);
IMPLEMENT_BINARY_FUNC_PP(pow);
Tensor *add(const Tensor *a, const Tensor *b) {
  while (google::_Check_string *_result = google::Check_EQImpl(google::GetReferenceableValue(a->state()),
                                                               google::GetReferenceableValue(b->state()),
                                                               "a->state()" " " "==" " " "b->state()"))
    google::LogMessageFatal("_file_name_", 257, google::CheckOpString(_result)).stream()
        << "tensors must be at same device";
  switch (a->state()) {
    case stensor::CPU:
      if (a->ShapeEquals(b)) {
        bool require_grad = (a->require_grad() || b->require_grad());
        Tensor *out_tensor = new Tensor(a->shape(), -1, require_grad);
        stensor::cpu_add(a->size(), a->cpu_data(), b->cpu_data(),
                         out_tensor->mutable_cpu_data());
        return out_tensor;
      } else {
        std::vector<int> shape_a;
        VecUint32ToVecInt(a->shape(), shape_a);
        std::vector<int> shape_b;
        VecUint32ToVecInt(b->shape(), shape_b);
        const std::vector<int> shape_out = stensor::broadcast(shape_a, shape_b);
        std::vector<uint32_t> shape_out_u;
        VecIntToUint32(shape_out, shape_out_u);
        bool require_grad = (a->require_grad() || b->require_grad());
        Tensor *out_tensor = new Tensor(shape_out_u, -1, require_grad);
        stensor::cpu_add_broadcast(a->cpu_data(), b->cpu_data(),
                                   shape_a, shape_b,
                                   out_tensor->mutable_cpu_data());
        return out_tensor;
      }
      break;
    case stensor::GPU:CHECK_EQ(a->device(), b->device()) << "tensors must be at same device";
      if (a->ShapeEquals(b)) {
        bool require_grad = (a->require_grad() || b->require_grad());
        Tensor *out_tensor = new Tensor(a->shape(), a->device(), require_grad);
        stensor::gpu_add(a->size(), a->gpu_data(), b->gpu_data(),
                         out_tensor->mutable_gpu_data());
        return out_tensor;
      } else {
        std::vector<int> shape_a;
        VecUint32ToVecInt(a->shape(), shape_a);
        std::vector<int> shape_b;
        VecUint32ToVecInt(b->shape(), shape_b);
        const std::vector<int> shape_out = stensor::broadcast(shape_a, shape_b);
        std::vector<uint32_t> shape_out_u;
        VecIntToUint32(shape_out, shape_out_u);
        bool require_grad = (a->require_grad() || b->require_grad());
        Tensor *out_tensor = new Tensor(shape_out_u, a->device(), require_grad);
        gpu_add_broadcast<float>(a->gpu_data(), b->gpu_data(),
                                 shape_a, shape_b,
                                 out_tensor->mutable_gpu_data());
        return out_tensor;
      }
      break;
  }
}

Tensor *matmul(const Tensor *a, const Tensor *b, int axis, bool transA, bool transB) {
  CHECK_EQ(a->state(), b->state()) << "tensors must be at same device";
  // inference shape
  uint32_t start_axis_a = a->CanonicalAxisIndex(axis);
  uint32_t start_axis_b = a->CanonicalAxisIndex(axis);
  int Ma = a->count(0, start_axis_a);
  int Na = a->count(start_axis_a, a->num_axes());
  if (transA) swap(Ma, Na);

  int Mb = b->count(0, start_axis_b);
  int Nb = b->count(start_axis_b, b->num_axes());
  if (transB) swap(Mb, Nb);

  std::vector<uint32_t> out_shape;
  for (int i = 0; i < start_axis_a; ++i)
    out_shape.push_back(a->shape(i));
  for (int i = start_axis_b; i < b->num_axes(); ++i)
    out_shape.push_back(b->shape(i));

  CHECK_EQ(Na, Mb) << "Shape mismatch";
  const CBLAS_TRANSPOSE TranA = transA ? CblasTrans : CblasNoTrans;
  const CBLAS_TRANSPOSE TranB = transB ? CblasTrans : CblasNoTrans;
  Tensor *out_tensor;
  bool require_grad = (a->require_grad() || b->require_grad());

  switch (a->state()) {
    case stensor::CPU:out_tensor = new Tensor(out_shape, -1, require_grad);
      stensor::cpu_gemm(TranA, TranB, Ma, Nb, Mb,
                        1.0f, a->cpu_data(), b->cpu_data(),
                        0.0f, out_tensor->mutable_cpu_data());
      return out_tensor;
      break;
    case stensor::GPU:CHECK_EQ(a->device(), b->device()) << "tensors must be at same device";
      out_tensor = new Tensor(out_shape, a->device(), require_grad);
      stensor::gpu_gemm(TranA, TranB, Ma, Nb, Mb,
                        1.0f, a->gpu_data(), b->gpu_data(),
                        0.0f, out_tensor->mutable_gpu_data());
      return out_tensor;
      break;
  }
}

Tensor *maximum(const Tensor *a, const Tensor *b) {
  CHECK_EQ(a->state(), b->state()) << "tensors must be at same device";
  CHECK(a->ShapeEquals(b)) << "tensors must be at same shape";
  bool require_grad = (a->require_grad() || b->require_grad());
  Tensor *out_tensor;
  switch (a->state()) {
    case stensor::CPU:out_tensor = new Tensor(a->shape(), -1, require_grad);
      stensor::cpu_maximum(a->size(), a->cpu_data(), b->cpu_data(),
                           out_tensor->mutable_cpu_data());
      return out_tensor;
      break;
    case stensor::GPU:CHECK_EQ(a->device(), b->device())
    << "tensors must be at same device";
      out_tensor = new Tensor(a->shape(), a->device(), require_grad);
      stensor::gpu_maximum(a->size(), a->gpu_data(), b->gpu_data(),
                           out_tensor->mutable_gpu_data());
      return out_tensor;
      break;
  }
}

Tensor *minimum(const Tensor *a, const Tensor *b) {
  CHECK_EQ(a->state(), b->state()) << "tensors must be at same device";
  CHECK(a->ShapeEquals(b)) << "tensors must be at same shape";
  bool require_grad = (a->require_grad() || b->require_grad());
  Tensor *out_tensor;
  switch (a->state()) {
    case stensor::CPU:out_tensor = new Tensor(a->shape(), -1, require_grad);
      stensor::cpu_minimum(a->size(), a->cpu_data(), b->cpu_data(),
                           out_tensor->mutable_cpu_data());
      return out_tensor;
      break;
    case stensor::GPU:CHECK_EQ(a->device(), b->device())
        << "tensors must be at same device";
      out_tensor = new Tensor(a->shape(), a->device(), require_grad);
      stensor::gpu_minimum(a->size(), a->gpu_data(), b->gpu_data(),
                           out_tensor->mutable_gpu_data());
      return out_tensor;
      break;
  }
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
                                     new_t->mutable_cpu_data());
      break;
    case GPU:
      gpu_rng_uniform<Tensor::Dtype>(new_t->size(),
                                     Tensor::Dtype(a),
                                     Tensor::Dtype(b),
                                     new_t->mutable_gpu_data());
      break;
    default:
      cpu_rng_uniform<Tensor::Dtype>(new_t->size(),
                                     Tensor::Dtype(a),
                                     Tensor::Dtype(b),
                                     new_t->mutable_cpu_data());
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
                                      new_t->mutable_cpu_data());
      break;
    case GPU:
      gpu_rng_gaussian<Tensor::Dtype>(new_t->size(),
                                      Tensor::Dtype(mu),
                                      Tensor::Dtype(sigma),
                                      new_t->mutable_gpu_data());
      break;
    default:
      cpu_rng_gaussian<Tensor::Dtype>(new_t->size(),
                                      Tensor::Dtype(mu),
                                      Tensor::Dtype(sigma),
                                      new_t->mutable_cpu_data());
      break;
  }
  return new_t;
}

}