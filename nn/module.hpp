/**
* Copyright 2021 wss
* Created by wss on 11月,21, 2021
*/
#ifndef STENSOR_NN_MODULE_HPP_
#define STENSOR_NN_MODULE_HPP_
#include "tensor.hpp"
#include "common.hpp"
#include "math_tesnsor.hpp"

namespace stensor {

namespace nn {
typedef shared_ptr<stensor::Tensor> SharedTensor;
typedef std::vector<shared_ptr<stensor::Tensor>> TensorVec;
class Module {
 public:
  virtual TensorVec forward(TensorVec& inputs) = 0;
  virtual void backward() = 0;
};
}//namespace nn

}//namespace stensor
#endif //STENSOR_NN_MODULE_HPP_
