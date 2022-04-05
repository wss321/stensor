/**
* Copyright 2021 wss
* Created by wss on 11月,16, 2021
*/
#ifndef STENSOR_CORE_MATH_TESNSOR_HPP_
#define STENSOR_CORE_MATH_TESNSOR_HPP_
#include "public/common.hpp"
#include "tensor.hpp"
#include "transpose.hpp"

namespace stensor {

/* math of Tensor */

/* self-op start*/
Tensor *sigmoid(const Tensor *tensor, Tensor *out = nullptr, bool grad_op = false);
Tensor *tanh(const Tensor *tensor, Tensor *out = nullptr, bool grad_op = false);
Tensor *relu(const Tensor *tensor, Tensor *out = nullptr, bool grad_op = false);
Tensor *elu(const Tensor *tensor, Tensor *out = nullptr, bool grad_op = false);
Tensor *gelu(const Tensor *tensor, Tensor *out = nullptr, bool grad_op = false);
Tensor *leakyrelu(const Tensor *tensor, Tensor *out = nullptr, bool grad_op = false);
Tensor *sign(const Tensor *tensor, Tensor *out = nullptr, bool grad_op = false);
Tensor *abs(const Tensor *tensor, Tensor *out = nullptr, bool grad_op = false);
Tensor *sqrt(const Tensor *tensor, Tensor *out = nullptr, bool grad_op = false);
Tensor *square(const Tensor *tensor, Tensor *out = nullptr, bool grad_op = false);
Tensor *clamp(const Tensor *tensor, float minVal, float maxVal, Tensor *out = nullptr, bool grad_op = false);
Tensor *repeat(const Tensor *tensor, int axis, int num, Tensor *out = nullptr, bool grad_op = false);
Tensor *softmax(const Tensor *tensor, int axis, Tensor *out = nullptr, bool grad_op = false);

Tensor *one_hot(const Tensor *tensor, int num_class, Tensor *out = nullptr, bool grad_op = false);
Tensor *argmax(const Tensor *tensor, int axis, Tensor *out = nullptr, bool grad_op = false);
/* self-op end*/

/* Tensor-scalar start*/

void set(Tensor *tensor, float val, bool grad_op = false);

Tensor *add(const Tensor *tensor, float val, Tensor *out = nullptr, bool grad_op = false);
Tensor *scale(const Tensor *tensor, float val, Tensor *out = nullptr, bool grad_op = false);
Tensor *pow(const Tensor *tensor, float val, Tensor *out = nullptr, bool grad_op = false);
Tensor *exp(const Tensor *tensor, Tensor *out = nullptr, bool grad_op = false);

/* Tensor-scalar end*/

/* Tensor-Tensor start*/
Tensor *add(const Tensor *a, const Tensor *b, Tensor *out = nullptr, bool grad_op = false);
Tensor *sub(const Tensor *a, const Tensor *b, Tensor *out = nullptr, bool grad_op = false);
Tensor *mul(const Tensor *a, const Tensor *b, Tensor *out = nullptr, bool grad_op = false);
Tensor *div(const Tensor *a, const Tensor *b, Tensor *out = nullptr, bool grad_op = false);
Tensor *pow(const Tensor *a, const Tensor *b, Tensor *out = nullptr, bool grad_op = false);

// matmul at last two axis
Tensor *matmul(const Tensor *a, const Tensor *b, int axis = -1,
               bool transA = false, bool transB = false, float beta = 0.0,
               Tensor *out = nullptr, bool grad_op = false);
Tensor *maximum(const Tensor *a, const Tensor *b, Tensor *out = nullptr, bool grad_op = false);
Tensor *minimum(const Tensor *a, const Tensor *b, Tensor *out = nullptr, bool grad_op = false);
Tensor *concat(const std::vector<Tensor *> &inputs, int axis, Tensor *out = nullptr);
/* Tensor-Tensor end*/

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
  Tensor *new_t = new Tensor(shape, device_id, require_grad);
  set(new_t, val);
  return new_t;
}

inline Tensor *zeros(const Tensor::ShapeType &shape, int device_id = -1, bool require_grad = false) {
  return constants(shape, 0.0, device_id, require_grad);
}

inline Tensor *ones(const Tensor::ShapeType &shape, int device_id = -1, bool require_grad = false) {
  return constants(shape, 1.0, device_id, require_grad);
}

inline Tensor *constants_like(Tensor *other, Tensor::Dtype val, bool require_grad = false) {
  Tensor *new_t = new Tensor(other->shape(),other->device(), require_grad);
  set(new_t, val);
  return new_t;
}

inline Tensor *zeros_like(Tensor *other, bool require_grad = false) {
  return constants_like(other, 0.0, require_grad);
}

inline Tensor *ones_like(Tensor *other, bool require_grad = false) {
  return constants_like(other, 1.0, require_grad);
}

/* Tensor Generator end*/

/*reduction*/
Tensor *sum(const Tensor *a, int dim, Tensor *out = nullptr, bool grad_op = false);
Tensor *mean(const Tensor *a, int dim, Tensor *out = nullptr, bool grad_op = false);
Tensor *var(const Tensor *a, int dim, Tensor *out = nullptr, bool grad_op = false, bool unbiased = true);
Tensor *std(const Tensor *a, int dim, Tensor *out = nullptr, bool grad_op = false, bool unbiased = true);
Tensor *asum(const Tensor *a, int dim, Tensor *out = nullptr, bool grad_op = false);

Tensor *sum(const Tensor *a, std::vector<int> dim, Tensor *out = nullptr, bool grad_op = false);
Tensor *mean(const Tensor *a, std::vector<int> dim, Tensor *out = nullptr, bool grad_op = false);
Tensor *var(const Tensor *a, std::vector<int> dim, Tensor *out = nullptr, bool grad_op = false, bool unbiased = true);
Tensor *std(const Tensor *a, std::vector<int> dim, Tensor *out = nullptr, bool grad_op = false, bool unbiased = true);
Tensor *asum(const Tensor *a, std::vector<int> dim, Tensor *out = nullptr, bool grad_op = false);
}

#endif //STENSOR_CORE_MATH_TESNSOR_HPP_
