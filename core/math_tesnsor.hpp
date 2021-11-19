/**
* Copyright 2021 wss
* Created by wss on 11月,16, 2021
*/
#ifndef STENSOR_CORE_MATH_TESNSOR_HPP_
#define STENSOR_CORE_MATH_TESNSOR_HPP_
#include "public/common.hpp"
#include "tensor.hpp"
#include "math_base_cpu.hpp"
#include "math_base_cuda.hpp"

namespace stensor {

/* math of Tensor */

/* self-op start*/
Tensor *sigmoid(Tensor *tensor, bool inplace = false);
Tensor *tanh(Tensor *tensor, bool inplace = false);
Tensor *relu(Tensor *tensor, bool inplace = false);
Tensor *elu(Tensor *tensor, bool inplace = false);
Tensor *gelu(Tensor *tensor, bool inplace = false);
Tensor *leakyrelu(Tensor *tensor, bool inplace = false);
Tensor *sign(Tensor *tensor, bool inplace = false);
Tensor *abs(Tensor *tensor, bool inplace = false);
Tensor *sqrt(Tensor *tensor, bool inplace = false);
Tensor *square(Tensor *tensor, bool inplace = false);
Tensor *clamp(Tensor *tensor, float minVal, float maxVal, bool inplace = false);

/* self-op end*/

/* Tensor-scalar start*/
void set(Tensor &tensor, const float val);
void set(Tensor *tensor, const float val);

Tensor *add(Tensor *tensor, const float val, bool inplace = false);
Tensor *scale(Tensor *tensor, const float val, bool inplace = false);
Tensor *pow(Tensor *tensor, const float val, bool inplace = false);
Tensor *exp(Tensor *tensor, bool inplace = false);

inline Tensor *add(Tensor &tensor, const float val, bool inplace = false) {
  return add(&tensor, val, inplace);
}

inline Tensor *scale(Tensor &tensor, const float val, bool inplace = false) {
  return scale(&tensor, val, inplace);
}

inline Tensor *pow(Tensor &tensor, const float val, bool inplace = false) {
  return pow(&tensor, val, inplace);
}
inline Tensor *exp(Tensor &tensor, bool inplace = false) {
  return exp(&tensor, inplace);
}

/* Tensor-scalar end*/

/* Tensor-Tensor start*/
Tensor *add(const Tensor *a, const Tensor *b);
Tensor *sub(const Tensor *a, const Tensor *b);
Tensor *mul(const Tensor *a, const Tensor *b);
Tensor *div(const Tensor *a, const Tensor *b);
Tensor *pow(const Tensor *a, const Tensor *b);

inline Tensor *add(const Tensor &a, const Tensor &b) { return add(&a, &b); }
inline Tensor *sub(const Tensor &a, const Tensor &b) { return sub(&a, &b); }
inline Tensor *mul(const Tensor &a, const Tensor &b) { return mul(&a, &b); }
inline Tensor *div(const Tensor &a, const Tensor &b) { return div(&a, &b); }
inline Tensor *pow(const Tensor &a, const Tensor &b) { return pow(&a, &b); }

// matmul at last two axis
Tensor *matmul(const Tensor *a, const Tensor *b, int axis = -1, bool transA = false, bool transB = false);
Tensor *maximum(const Tensor *a, const Tensor *b);
Tensor *minimum(const Tensor *a, const Tensor *b);

/* Tensor-Tensor start*/

/* math of Tensor end */

/* Tensor Generator*/
Tensor *random(const Tensor::ShapeType &shape, float a, float b, int device_id = -1, bool require_grad = false);
inline Tensor *random(const Tensor::ShapeType &shape, int device_id = -1, bool require_grad = false) {
  return random(shape, 0.0, 1.0, device_id, require_grad);
}

Tensor *random_gaussian(const Tensor::ShapeType &shape,
                        float mu,
                        float sigma,
                        int device_id = -1,
                        bool require_grad = false);

inline Tensor *random_gaussian(const Tensor::ShapeType &shape, int device_id = -1, bool require_grad = false) {
  return random_gaussian(shape, 0.0, 1.0, device_id, require_grad);
}

inline Tensor *constants(const Tensor::ShapeType &shape,
                         Tensor::Dtype val,
                         int device_id = -1,
                         bool require_grad = false) {
  stensor::Tensor *new_t = new Tensor(shape, device_id, require_grad);
  stensor::set(new_t, val);
  return new_t;
}

inline Tensor *zeros(const Tensor::ShapeType &shape, int device_id = -1, bool require_grad = false) {
  return constants(shape, 0.0, device_id, require_grad);
}

inline Tensor *ones(const Tensor::ShapeType &shape, int device_id = -1, bool require_grad = false) {
  return constants(shape, 1.0, device_id, require_grad);
}

inline Tensor *constants_like(Tensor *other, Tensor::Dtype val, bool require_grad = false) {
  Tensor *new_t = new Tensor(other, require_grad);
  stensor::set(new_t, val);
  return new_t;
}

inline Tensor *zeros_like(Tensor *other, bool require_grad = false) {
  return constants_like(other, 0.0, require_grad);
}

inline Tensor *ones_like(Tensor *other, bool require_grad = false) {
  return constants_like(other, 1.0, require_grad);
}

/* Tensor Generator end*/
}

#endif //STENSOR_CORE_MATH_TESNSOR_HPP_
