/**
* Copyright 2021 wss
* Created by wss on 11月,16, 2021
*/
#ifndef STENSOR_CORE_MATH_TESNSOR_HPP_
#define STENSOR_CORE_MATH_TESNSOR_HPP_
#include "public/common.hpp"
#include "tensor.hpp"

namespace stensor {

/* math of Tensor */
void set(Tensor &tensor, const float val);
void set(Tensor *tensor, const float val);

Tensor *add(Tensor *tensor, const float val, bool inplace = false);
Tensor *sub(Tensor *tensor, const float val, bool inplace = false);
Tensor *scale(Tensor *tensor, const float val, bool inplace = false);
Tensor *pow(Tensor *tensor, const float val, bool inplace = false);
Tensor *exp(Tensor *tensor, bool inplace = false);

inline Tensor *add(Tensor &tensor, const float val, bool inplace = false) {
  return add(&tensor, val, inplace);
}
inline Tensor *sub(Tensor &tensor, const float val, bool inplace = false) {
  return sub(&tensor, val, inplace);
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

Tensor *add(const Tensor *a, const Tensor *b);
Tensor *sub(const Tensor *a, const Tensor *b);
Tensor *mul(const Tensor *a, const Tensor *b);
Tensor *div(const Tensor *a, const Tensor *b);

inline Tensor *add(const Tensor &a, const Tensor &b) { return add(&a, &b); }
inline Tensor *sub(const Tensor &a, const Tensor &b) { return sub(&a, &b); }
inline Tensor *mul(const Tensor &a, const Tensor &b) { return mul(&a, &b); }
inline Tensor *div(const Tensor &a, const Tensor &b) { return div(&a, &b); }
/* math of Tensor end */

/* Tensor Generator*/
Tensor *random(const Tensor::ShapeType &shape, bool require_grad, float a, float b);
Tensor *random(const Tensor::ShapeType &shape, bool require_grad = false);
Tensor *random(const Tensor::ShapeType &shape, float a, float b);

Tensor *random_gaussian(const Tensor::ShapeType &shape, bool require_grad, float mu, float sigma);
Tensor *random_gaussian(const Tensor::ShapeType &shape, bool require_grad = false);
Tensor *random_gaussian(const Tensor::ShapeType &shape, float mu, float sigma);

Tensor *constants(const Tensor::ShapeType &shape, Tensor::Dtype val, bool require_grad = false);
Tensor *zeros(const Tensor::ShapeType &shape, bool require_grad = false);
Tensor *ones(const Tensor::ShapeType &shape, bool require_grad = false);
Tensor *zeros_like(Tensor *other, bool require_grad = false);
Tensor *ones_like(Tensor *other, bool require_grad = false);
Tensor *constants_like(Tensor *other, Tensor::Dtype val, bool require_grad = false);
/* Tensor Generator end*/
}

#endif //STENSOR_CORE_MATH_TESNSOR_HPP_
