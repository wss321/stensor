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

/* self-op start*/
template<typename Dtype>
Tensor<Dtype> *sigmoid(const Tensor<Dtype> *tensor, Tensor<Dtype> *out = nullptr, bool grad_op = false);

template<typename Dtype>
Tensor<Dtype> *tanh(const Tensor<Dtype> *tensor, Tensor<Dtype> *out = nullptr, bool grad_op = false);

template<typename Dtype>
Tensor<Dtype> *relu(const Tensor<Dtype> *tensor, Tensor<Dtype> *out = nullptr, bool grad_op = false);

template<typename Dtype>
Tensor<Dtype> *elu(const Tensor<Dtype> *tensor, Tensor<Dtype> *out = nullptr, bool grad_op = false);

template<typename Dtype>
Tensor<Dtype> *gelu(const Tensor<Dtype> *tensor, Tensor<Dtype> *out = nullptr, bool grad_op = false);

template<typename Dtype>
Tensor<Dtype> *leakyrelu(const Tensor<Dtype> *tensor, Tensor<Dtype> *out = nullptr, bool grad_op = false);

template<typename Dtype>
Tensor<Dtype> *sign(const Tensor<Dtype> *tensor, Tensor<Dtype> *out = nullptr, bool grad_op = false);

template<typename Dtype>
Tensor<Dtype> *abs(const Tensor<Dtype> *tensor, Tensor<Dtype> *out = nullptr, bool grad_op = false);

template<typename Dtype>
Tensor<Dtype> *sqrt(const Tensor<Dtype> *tensor, Tensor<Dtype> *out = nullptr, bool grad_op = false);

template<typename Dtype>
Tensor<Dtype> *square(const Tensor<Dtype> *tensor, Tensor<Dtype> *out = nullptr, bool grad_op = false);

template<typename Dtype>
Tensor<Dtype> *clamp(const Tensor<Dtype> *tensor,
                     float minVal,
                     float maxVal,
                     Tensor<Dtype> *out = nullptr,
                     bool grad_op = false);

template<typename Dtype>
Tensor<Dtype> *repeat(const Tensor<Dtype> *tensor,
                      int axis,
                      int num,
                      Tensor<Dtype> *out = nullptr,
                      bool grad_op = false);

template<typename Dtype>
Tensor<Dtype> *softmax(const Tensor<Dtype> *tensor, int axis, Tensor<Dtype> *out = nullptr, bool grad_op = false);

template<typename Dtype>
Tensor<Dtype> *one_hot(const Tensor<Dtype> *tensor, int num_class, Tensor<Dtype> *out = nullptr, bool grad_op = false);
template<typename Dtype>

Tensor<Dtype> *argmax(const Tensor<Dtype> *tensor, int axis, Tensor<Dtype> *out = nullptr, bool grad_op = false);
/* self-op end*/

/* Tensor-scalar start*/
template<typename Dtype>
void set(Tensor<Dtype> *tensor, float val, bool grad_op = false);

template<typename Dtype>
Tensor<Dtype> *add(const Tensor<Dtype> *tensor, float val, Tensor<Dtype> *out = nullptr, bool grad_op = false);

template<typename Dtype>
Tensor<Dtype> *scale(const Tensor<Dtype> *tensor, float val, Tensor<Dtype> *out = nullptr, bool grad_op = false);

template<typename Dtype>
Tensor<Dtype> *pow(const Tensor<Dtype> *tensor, float val, Tensor<Dtype> *out = nullptr, bool grad_op = false);

template<typename Dtype>
Tensor<Dtype> *exp(const Tensor<Dtype> *tensor, Tensor<Dtype> *out = nullptr, bool grad_op = false);

/* Tensor-scalar end*/

/* Tensor-Tensor start*/
template<typename Dtype>
Tensor<Dtype> *add(const Tensor<Dtype> *a, const Tensor<Dtype> *b, Tensor<Dtype> *out = nullptr, bool grad_op = false);

template<typename Dtype>
Tensor<Dtype> *sub(const Tensor<Dtype> *a, const Tensor<Dtype> *b, Tensor<Dtype> *out = nullptr, bool grad_op = false);

template<typename Dtype>
Tensor<Dtype> *mul(const Tensor<Dtype> *a, const Tensor<Dtype> *b, Tensor<Dtype> *out = nullptr, bool grad_op = false);

template<typename Dtype>
Tensor<Dtype> *div(const Tensor<Dtype> *a, const Tensor<Dtype> *b, Tensor<Dtype> *out = nullptr, bool grad_op = false);

template<typename Dtype>
Tensor<Dtype> *pow(const Tensor<Dtype> *a, const Tensor<Dtype> *b, Tensor<Dtype> *out = nullptr, bool grad_op = false);

// matmul at last two axis
template<typename Dtype>
Tensor<Dtype> *matmul(const Tensor<Dtype> *a, const Tensor<Dtype> *b, int axis = -1,
                      bool transA = false, bool transB = false, float beta = 0.0,
                      Tensor<Dtype> *out = nullptr, bool grad_op = false);
template<typename Dtype>
Tensor<Dtype> *maximum(const Tensor<Dtype> *a,
                       const Tensor<Dtype> *b,
                       Tensor<Dtype> *out = nullptr,
                       bool grad_op = false);
template<typename Dtype>
Tensor<Dtype> *minimum(const Tensor<Dtype> *a,
                       const Tensor<Dtype> *b,
                       Tensor<Dtype> *out = nullptr,
                       bool grad_op = false);
template<typename Dtype>
Tensor<Dtype> *concat(const std::vector<Tensor<Dtype> *> &inputs,int axis, Tensor<Dtype>* out = nullptr);
/* Tensor-Tensor end*/

/* math of Tensor end */

/* Tensor Generator*/
template<typename Dtype>
Tensor<Dtype> *random(const std::vector<int> &shape, float a, float b, int device_id = -1, bool require_grad = false);

template<typename Dtype>
inline Tensor<Dtype> *random(const std::vector<int> &shape, int device_id = -1, bool require_grad = false) {
  return random<Dtype>(shape, 0.0, 1.0, device_id, require_grad);
}

template<typename Dtype>
Tensor<Dtype> *random_gaussian(const std::vector<int> &shape,
                               float mu,
                               float sigma,
                               int device_id = -1,
                               bool require_grad = false);
template<typename Dtype>
inline Tensor<Dtype> *random_gaussian(const std::vector<int> &shape, int device_id = -1, bool require_grad = false) {
  return random_gaussian<Dtype>(shape, 0.0, 1.0, device_id, require_grad);
}

template<typename Dtype>
inline Tensor<Dtype> *constants(const std::vector<int> &shape,
                                Dtype val,
                                int device_id = -1,
                                bool require_grad = false) {
  Tensor<Dtype> *new_t = new Tensor<Dtype>(shape, device_id, require_grad);
  set(new_t, val);
  return new_t;
}

template<typename Dtype>
inline Tensor<Dtype> *zeros(const std::vector<int> &shape, int device_id = -1, bool require_grad = false) {
  return constants<Dtype>(shape, 0.0, device_id, require_grad);
}

template<typename Dtype>
inline Tensor<Dtype> *ones(const std::vector<int> &shape, int device_id = -1, bool require_grad = false) {
  return constants<Dtype>(shape, 1.0, device_id, require_grad);
}
template<typename Dtype>
inline Tensor<Dtype> *constants_like(Tensor<Dtype> *other, Dtype val, bool require_grad = false) {
  Tensor<Dtype> *new_t = new Tensor<Dtype>(other->shape(), other->device(), require_grad);
  set(new_t, val);
  return new_t;
}

template<typename Dtype>
inline Tensor<Dtype> *zeros_like(Tensor<Dtype> *other, bool require_grad = false) {
  return constants_like<Dtype>(other, 0.0, require_grad);
}

template<typename Dtype>
inline Tensor<Dtype> *ones_like(Tensor<Dtype> *other, bool require_grad = false) {
  return constants_like<Dtype>(other, 1.0, require_grad);
}

/* Tensor Generator end*/

/*reduction*/
template<typename Dtype>
Tensor<Dtype> *sum(const Tensor<Dtype> *a, int axis, Tensor<Dtype> *out = nullptr, bool grad_op = false);
template<typename Dtype>
Tensor<Dtype> *mean(const Tensor<Dtype> *a, int axis, Tensor<Dtype> *out = nullptr, bool grad_op = false);
template<typename Dtype>
Tensor<Dtype> *asum(const Tensor<Dtype> *a, int axis, Tensor<Dtype> *out = nullptr, bool grad_op = false);
}

#endif //STENSOR_CORE_MATH_TESNSOR_HPP_
