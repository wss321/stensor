/**
* Copyright 2021 wss
* Created by wss on 11月,16, 2021
*/
#include "math_tesnsor.hpp"
#include "math_base.hpp"
namespace stensor {
/* math of Tensor */
//TODO:GPU Mode
void set(Tensor &tensor, float val) {
  float *data = tensor.mutable_cpu_data();
  stensor::set(tensor.size(), val, data);
}

void set(Tensor *tensor, float val) {
  float *data = tensor->mutable_cpu_data();
  stensor::set(tensor->size(), val, data);
}

Tensor *add(Tensor *tensor, float val, bool inplace) {
  if (inplace) {
    float *data = tensor->mutable_cpu_data();
    stensor::add(tensor->size(), data, val, data);
    return tensor;
  } else {
    const float *data = tensor->cpu_data();
    Tensor *out_tensor = new Tensor(tensor, tensor->require_grad());
    float *out_data = out_tensor->mutable_cpu_data();
    stensor::add(out_tensor->size(), data, val, out_data);
    return out_tensor;
  }
}

Tensor *sub(Tensor *tensor, float val, bool inplace) {
  if (inplace) {
    float *data = tensor->mutable_cpu_data();
    stensor::sub(tensor->size(), data, val, data);
    return tensor;
  } else {
    const float *data = tensor->cpu_data();
    Tensor *out_tensor = new Tensor(tensor, tensor->require_grad());
    float *out_data = out_tensor->mutable_cpu_data();
    stensor::sub(out_tensor->size(), data, val, out_data);
    return out_tensor;
  }
}
Tensor *scale(Tensor *tensor, float val, bool inplace) {
  if (inplace) {
    float *data = tensor->mutable_cpu_data();
    stensor::scale(tensor->size(), data, val, data);
    return tensor;
  } else {
    const float *data = tensor->cpu_data();
    Tensor *out_tensor = new Tensor(tensor, tensor->require_grad());
    float *out_data = out_tensor->mutable_cpu_data();
    stensor::scale(out_tensor->size(), data, val, out_data);
    return out_tensor;
  }
}

Tensor *pow(Tensor *tensor, float val, bool inplace) {
  if (inplace) {
    float *data = tensor->mutable_cpu_data();
    stensor::pow(tensor->size(), data, val, data);
    return tensor;
  } else {
    const float *data = tensor->cpu_data();
    Tensor *out_tensor = new Tensor(tensor, tensor->require_grad());
    float *out_data = out_tensor->mutable_cpu_data();
    stensor::pow(out_tensor->size(), data, val, out_data);
    return out_tensor;
  }
}

Tensor *exp(Tensor *tensor, bool inplace) {
  if (inplace) {
    float *data = tensor->mutable_cpu_data();
    stensor::exp(tensor->size(), data, data);
    return tensor;
  } else {
    const float *data = tensor->cpu_data();
    Tensor *out_tensor = new Tensor(tensor, tensor->require_grad());
    float *out_data = out_tensor->mutable_cpu_data();
    stensor::exp(out_tensor->size(), data, out_data);
    return out_tensor;
  }
}

// Tensor op Tensor
// TODO: op on GPU
inline int broadcast_index(const Tensor::ShapeType &shape_in, const Tensor::ShapeType &shape_out, int index) {
  std::vector<int> indices_in_result(shape_out.size(), 0);
  indices_in_result[shape_out.size() - 1] = index % static_cast<int>(shape_out[shape_out.size() - 1]);
  int div = 1;
  for (int i = shape_out.size() - 2; i >= 0; --i) {
    div *=  static_cast<int>(shape_out[i + 1]);
    indices_in_result[i] = (index / div)% static_cast<int>(shape_out[i]);
  }

  int out = 0;
  for (int i = 0; i < indices_in_result.size(); ++i) {
    int m = std::min(indices_in_result[i], static_cast<int>(shape_in[i]));
    out *= static_cast<int>(shape_in[i]);
    if (shape_in[i]!=1)
      out += m;
  }
  return out;
}
// TODO: op on GPU
Tensor *add(const Tensor *a, const Tensor *b) {
  if (a->shape() == b->shape()) {
    bool require_grad = (a->require_grad() || b->require_grad());
    Tensor *out_tensor = new Tensor(a->shape(), require_grad);
    stensor::add(a->size(), a->cpu_data(), b->cpu_data(), out_tensor->mutable_cpu_data());
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
      *out_data = a->data_at(index_a) + b->data_at(index_b);
      out_data ++ ;
    }
    return out_tensor;
  }
}
// TODO: op on GPU
Tensor *sub(const Tensor *a, const Tensor *b) {
  if (a->shape() == b->shape()) {
    bool require_grad = (a->require_grad() || b->require_grad());
    Tensor *out_tensor = new Tensor(a->shape(), require_grad);
    stensor::sub(a->size(), a->cpu_data(), b->cpu_data(), out_tensor->mutable_cpu_data());
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
      out_data ++ ;
    }
    return out_tensor;
  }
}
// TODO: op on GPU
Tensor *mul(const Tensor *a, const Tensor *b) {
  if (a->shape() == b->shape()) {
    bool require_grad = (a->require_grad() || b->require_grad());
    Tensor *out_tensor = new Tensor(a->shape(), require_grad);
    stensor::mul(a->size(), a->cpu_data(), b->cpu_data(), out_tensor->mutable_cpu_data());
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
      out_data ++ ;
    }
    return out_tensor;
  }
}

// TODO: op on GPU
Tensor *div(const Tensor *a, const Tensor *b) {
  if (a->shape() == b->shape()) {
    bool require_grad = (a->require_grad() || b->require_grad());
    Tensor *out_tensor = new Tensor(a->shape(), require_grad);
    stensor::div(a->size(), a->cpu_data(), b->cpu_data(), out_tensor->mutable_cpu_data());
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
      *out_data = a->data_at(index_a) / (b->data_at(index_b)+1e-8);
      out_data ++ ;
    }
    return out_tensor;
  }
}

/* math of Tensor end */

/* Tensor Generator*/

Tensor *random(const Tensor::ShapeType &shape, bool require_grad, float a, float b) {
  Tensor *new_t = new Tensor(shape, require_grad);
  stensor::rng_uniform<Tensor::Dtype>(new_t->size(), Tensor::Dtype(a), Tensor::Dtype(b), new_t->mutable_cpu_data());
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
  stensor::rng_gaussian<Tensor::Dtype>(new_t->size(),
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