/**
* Copyright 2021 wss
* Created by wss on 11月,26, 2021
*/
#ifndef STENSOR_NN_SIGMOID_LAYER_HPP_
#define STENSOR_NN_SIGMOID_LAYER_HPP_
#include "module.hpp"

namespace stensor {

namespace nn {
class Sigmoid : public Module {
 public:
  explicit Sigmoid(std::string name, int device=-1){
    name_ = name;
    type_ = "Sigmoid";
    if (device > -1) this->state_ = GPU;
    else this->state_ = CPU;
  };
  ~Sigmoid() {};
  TensorVec forward(TensorVec &inputs);
  void backward();
 private:
  void backward_cpu();
  void backward_gpu();
  SharedTensor result_;
 DISABLE_COPY_AND_ASSIGN(Sigmoid);
};
};//namespace nn
}// namespace stensor
#endif //STENSOR_NN_SIGMOID_LAYER_HPP_
