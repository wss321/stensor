/**
* Copyright 2021 wss
* Created by wss on 11月,16, 2021
*/
#include "math_tesnsor.hpp"
#include "math_base_cpu.hpp"
namespace stensor {
/* math of Tensor */
//TODO:GPU Mode
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

// Tensor op Tensor
// TODO: op on GPU
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
#define MIN(a, b, c) c = a>b ? b: a
#define BROADCAST_INDEX(index, n, sy, shape_a, shape_b, shape_y) \
  int indices_in_result[32]{0};\
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
    MIN(indices_in_result[i], static_cast<int>(shape_a[i]), ma); \
    int mb;        \
    MIN(indices_in_result[i], static_cast<int>(shape_b[i]), mb);  \
    index_a *= static_cast<int>(shape_a[i]); \
    index_b *= static_cast<int>(shape_b[i]); \
    if (shape_a[i]!=1)  index_a += ma; \
    if (shape_b[i]!=1)  index_b += mb; \
  }
// TODO: op on GPU
Tensor *add(const Tensor *a, const Tensor *b) {
  CHECK_EQ(a->state(), b->state())<<"tensors must be at same device";
  switch (a->state()) {
    case stensor::CPU:
      if (a->shape() == b->shape()) {
        bool require_grad = (a->require_grad() || b->require_grad());
        Tensor *out_tensor = new Tensor(a->shape(), require_grad);
        stensor::cpu_add(a->size(), a->cpu_data(), b->cpu_data(), out_tensor->mutable_cpu_data());
        return out_tensor;
      } else {//broadcast
        Tensor::ShapeType shape_a(a->shape());
        Tensor::ShapeType shape_b(b->shape());
        const Tensor::ShapeType shape_out = stensor::broadcast(shape_a, shape_b);

        bool require_grad = (a->require_grad() || b->require_grad());
        Tensor *out_tensor = new Tensor(shape_out, require_grad);
        Tensor::Dtype *out_data = out_tensor->mutable_cpu_data();
        for (int i = 0; i < out_tensor->size(); ++i) {
          BROADCAST_INDEX(i, out_tensor.size(), shape_out.size(), shape_a, shape_b, shape_out);
//          int index_a = broadcast_index(shape_a, shape_out, i);
//          int index_b = broadcast_index(shape_b, shape_out, i);
          *out_data = a->data_at(index_a) + b->data_at(index_b);
          out_data++;
        }
        return out_tensor;
      }
      break;
    case stensor::GPU:
      if (a->shape() == b->shape()) {
        bool require_grad = (a->require_grad() || b->require_grad());
        Tensor *out_tensor = new Tensor(a->shape(), require_grad, a->device());
        stensor::gpu_add(a->size(), a->gpu_data(), b->gpu_data(), out_tensor->mutable_gpu_data());
        return out_tensor;
      } else {
        Tensor::ShapeType shape_a(a->shape());
        Tensor::ShapeType shape_b(b->shape());
        bool require_grad = (a->require_grad() || b->require_grad());
        Tensor *out_tensor = new Tensor(a->shape(), require_grad, a->device());
        gpu_add_broadcast<float>(a->gpu_data(), b->gpu_data(), shape_a, shape_b, out_tensor->mutable_gpu_data());
        return out_tensor;
      }
      break;
  }
}
// TODO: op on GPU
Tensor *sub(const Tensor *a, const Tensor *b) {
  if (a->shape() == b->shape()) {
    bool require_grad = (a->require_grad() || b->require_grad());
    Tensor *out_tensor = new Tensor(a->shape(), require_grad);
    stensor::cpu_sub(a->size(), a->cpu_data(), b->cpu_data(), out_tensor->mutable_cpu_data());
    return out_tensor;
  } else {
    Tensor::ShapeType shape_a(a->shape());
    Tensor::ShapeType shape_b(b->shape());
    const Tensor::ShapeType shape_out = stensor::broadcast(shape_a, shape_b);

    bool require_grad = (a->require_grad() || b->require_grad());
    Tensor *out_tensor = new Tensor(shape_out, require_grad);
    Tensor::Dtype *out_data = out_tensor->mutable_cpu_data();
    for (int i = 0; i < out_tensor->size(); ++i) {
      int index_a = broadcast_index(shape_a, shape_out, i);
      int index_b = broadcast_index(shape_b, shape_out, i);
      *out_data = a->data_at(index_a) - b->data_at(index_b);
      out_data++;
    }
    return out_tensor;
  }
}
// TODO: op on GPU
Tensor *mul(const Tensor *a, const Tensor *b) {
  if (a->shape() == b->shape()) {
    bool require_grad = (a->require_grad() || b->require_grad());
    Tensor *out_tensor = new Tensor(a->shape(), require_grad);
    stensor::cpu_mul(a->size(), a->cpu_data(), b->cpu_data(), out_tensor->mutable_cpu_data());
    return out_tensor;
  } else {
    Tensor::ShapeType shape_a(a->shape());
    Tensor::ShapeType shape_b(b->shape());
    const Tensor::ShapeType shape_out = stensor::broadcast(shape_a, shape_b);

    bool require_grad = (a->require_grad() || b->require_grad());
    Tensor *out_tensor = new Tensor(shape_out, require_grad);
    Tensor::Dtype *out_data = out_tensor->mutable_cpu_data();
    for (int i = 0; i < out_tensor->size(); ++i) {
      int index_a = broadcast_index(shape_a, shape_out, i);
      int index_b = broadcast_index(shape_b, shape_out, i);
      *out_data = a->data_at(index_a) * b->data_at(index_b);
      out_data++;
    }
    return out_tensor;
  }
}

// TODO: op on GPU
Tensor *div(const Tensor *a, const Tensor *b) {
  if (a->shape() == b->shape()) {
    bool require_grad = (a->require_grad() || b->require_grad());
    Tensor *out_tensor = new Tensor(a->shape(), require_grad);
    stensor::cpu_div(a->size(), a->cpu_data(), b->cpu_data(), out_tensor->mutable_cpu_data());
    return out_tensor;
  } else {
    Tensor::ShapeType shape_a(a->shape());
    Tensor::ShapeType shape_b(b->shape());
    const Tensor::ShapeType shape_out = stensor::broadcast(shape_a, shape_b);

    bool require_grad = (a->require_grad() || b->require_grad());
    Tensor *out_tensor = new Tensor(shape_out, require_grad);
    Tensor::Dtype *out_data = out_tensor->mutable_cpu_data();
    for (int i = 0; i < out_tensor->size(); ++i) {
      int index_a = broadcast_index(shape_a, shape_out, i);
      int index_b = broadcast_index(shape_b, shape_out, i);
      *out_data = a->data_at(index_a) / (b->data_at(index_b) + 1e-8);
      out_data++;
    }
    return out_tensor;
  }
}

/* math of Tensor end */

/* Tensor Generator*/

Tensor *random(const Tensor::ShapeType &shape, bool require_grad, float a, float b) {
  Tensor *new_t = new Tensor(shape, require_grad);
  stensor::cpu_rng_uniform<Tensor::Dtype>(new_t->size(), Tensor::Dtype(a), Tensor::Dtype(b), new_t->mutable_cpu_data());
  return new_t;
}

Tensor *random(const Tensor::ShapeType &shape, bool require_grad) {
  return random(shape, require_grad, 0.0, 1.0);
}
Tensor *random(const Tensor::ShapeType &shape, float a, float b) {
  return random(shape, false, a, b);
}

Tensor *random_gaussian(const Tensor::ShapeType &shape, bool require_grad, float mu, float sigma) {
  Tensor *new_t = new Tensor(shape, require_grad);
  stensor::cpu_rng_gaussian<Tensor::Dtype>(new_t->size(),
                                           Tensor::Dtype(mu),
                                           Tensor::Dtype(sigma),
                                           new_t->mutable_cpu_data());
  return new_t;
}

Tensor *random_gaussian(const Tensor::ShapeType &shape, bool require_grad) {
  return random_gaussian(shape, require_grad, 0.0, 1.0);
}
Tensor *random_gaussian(const Tensor::ShapeType &shape, float mu, float sigma) {
  return random_gaussian(shape, false, mu, sigma);
}

Tensor *constants(const Tensor::ShapeType &shape, Tensor::Dtype val, bool require_grad) {
  stensor::Tensor *new_t = new Tensor(shape, require_grad);
  stensor::set(new_t, val);
  return new_t;
}

Tensor *zeros(const Tensor::ShapeType &shape, bool require_grad) {
  return constants(shape, 0.0, require_grad);
}

Tensor *ones(const Tensor::ShapeType &shape, bool require_grad) {
  return constants(shape, 1.0, require_grad);
}
Tensor *constants_like(Tensor *other, Tensor::Dtype val, bool require_grad) {
  Tensor *new_t = new Tensor(other, require_grad);
  stensor::set(new_t, val);
  return new_t;
}

Tensor *zeros_like(Tensor *other, bool require_grad) {
  return constants_like(other, 0.0, require_grad);
}

Tensor *ones_like(Tensor *other, bool require_grad) {
  return constants_like(other, 1.0, require_grad);
}
}