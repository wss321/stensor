/**
* Copyright 2021 wss
* Created by wss on 11月,16, 2021
*/
#include "math_tesnsor.hpp"
#include "math_base_cpu.hpp"
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
inline int broadcast_index(const Tensor::ShapeType &shape_in, const Tensor::ShapeType &shape_out, int index) {
  std::vector<int> indices_in_result(shape_out.size(), 0);
  indices_in_result[shape_out.size() - 1] = index % static_cast<int>(shape_out[shape_out.size() - 1]);
  int div = 1;
  for (int i = shape_out.size() - 2; i >= 0; --i) {
    div *= static_cast<int>(shape_out[i + 1]);
    indices_in_result[i] = (index / div) % static_cast<int>(shape_out[i]);
  }

  int out = 0;
  for (int i = 0; i < indices_in_result.size(); ++i) {
    int m = std::min(indices_in_result[i], static_cast<int>(shape_in[i]));
    out *= static_cast<int>(shape_in[i]);
    if (shape_in[i] != 1)
      out += m;
  }
  return out;
}
#define MIN_FUNC(a, b, c) c = a>b ? b: a
#define BROADCAST_INDEX(index, n, sy, shape_a, shape_b, shape_y, index_a, index_b) \
  int indices_in_result[MAX_AXES]{0};\
  indices_in_result[sy - 1] = index % static_cast<int>(shape_y[sy - 1]); \
  int div = 1; \
  for (int i = sy - 2; i >= 0; --i) { \
    div *=  static_cast<int>(shape_y[i + 1]); \
    indices_in_result[i] = (index / div)% static_cast<int>(shape_y[i]); \
  } \
  int index_a = 0; \
  int index_b = 0; \
  for (int i = 0; i < sy; ++i) { \
    int ma;           \
    MIN_FUNC(indices_in_result[i], static_cast<int>(shape_a[i]), ma); \
    int mb;        \
    MIN_FUNC(indices_in_result[i], static_cast<int>(shape_b[i]), mb);  \
    index_a *= static_cast<int>(shape_a[i]); \
    index_b *= static_cast<int>(shape_b[i]); \
    if (shape_a[i]!=1)  index_a += ma; \
    if (shape_b[i]!=1)  index_b += mb; \
  }

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
      if (a->shape() == b->shape()) {\
        bool require_grad = (a->require_grad() || b->require_grad());\
        Tensor *out_tensor = new Tensor(a->shape(), -1, require_grad);\
        stensor::cpu_##name(a->size(), a->cpu_data(), b->cpu_data(), out_tensor->mutable_cpu_data());\
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
        stensor::cpu_##name##_broadcast(a->cpu_data(), b->cpu_data(), shape_a, shape_b, out_tensor->mutable_cpu_data());\
        return out_tensor;\
      }\
      break;\
    case stensor::GPU:\
      if (a->shape() == b->shape()) {\
        bool require_grad = (a->require_grad() || b->require_grad());\
        Tensor *out_tensor = new Tensor(a->shape(), a->device(), require_grad);\
        stensor::gpu_##name(a->size(), a->gpu_data(), b->gpu_data(), out_tensor->mutable_gpu_data());\
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
        gpu_##name##_broadcast<float>(a->gpu_data(), b->gpu_data(), shape_a, shape_b, out_tensor->mutable_gpu_data());\
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
      if (a->shape() == b->shape()) {
        bool require_grad = (a->require_grad() || b->require_grad());
        Tensor *out_tensor = new Tensor(a->shape(), -1,  require_grad);
        stensor::cpu_add(a->size(), a->cpu_data(), b->cpu_data(), out_tensor->mutable_cpu_data());
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
        stensor::cpu_add_broadcast(a->cpu_data(), b->cpu_data(), shape_a, shape_b, out_tensor->mutable_cpu_data());
        return out_tensor;
      }
      break;
    case stensor::GPU:
      if (a->shape() == b->shape()) {
        bool require_grad = (a->require_grad() || b->require_grad());
        Tensor *out_tensor = new Tensor(a->shape(), a->device(), require_grad);
        stensor::gpu_add(a->size(), a->gpu_data(), b->gpu_data(), out_tensor->mutable_gpu_data());
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
        gpu_add_broadcast<float>(a->gpu_data(), b->gpu_data(), shape_a, shape_b, out_tensor->mutable_gpu_data());
        return out_tensor;
      }
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