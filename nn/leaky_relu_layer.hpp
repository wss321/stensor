/**
* Copyright 2021 wss
* Created by wss on 11月,26, 2021
*/
#ifndef STENSOR_NN_LEAKY_RELU_LAYER_HPP_
#define STENSOR_NN_LEAKY_RELU_LAYER_HPP_
#include "module.hpp"
namespace stensor {

namespace nn {
class LeakyReLU : public Module {
 public:
  explicit LeakyReLU(std::string name, float alpha = 0.01, int device = -1, bool inplace = false) :
      inplace_(inplace) {
    alpha_ = alpha;
    name_ = name;
    type_ = "LeakyReLU";
    if (device > -1) this->state_ = GPU;
    else this->state_ = CPU;
  };
  ~LeakyReLU() {};
  TensorVec forward(TensorVec &inputs);
  void backward();
 private:
  void backward_cpu();
  void backward_gpu();
  bool inplace_;
  float alpha_;
  SharedTensor result_;
 DISABLE_COPY_AND_ASSIGN(LeakyReLU);
};
};//namespace nn
}

#endif //STENSOR_NN_LEAKY_RELU_LAYER_HPP_
