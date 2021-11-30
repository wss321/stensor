/**
* Copyright 2021 wss
* Created by wss on 11月,26, 2021
*/
#ifndef STENSOR_NN_RELU_LAYER_HPP_
#define STENSOR_NN_RELU_LAYER_HPP_
#include "module.hpp"
namespace stensor {

namespace nn {
class ReLU : public Module {
 public:
  explicit ReLU(std::string name, int device=-1, bool inplace = false) :
  inplace_(inplace) { name_ = name; type_ = "ReLU";
    if (device > -1) this->state_ = GPU;
    else this->state_ = CPU;
  };
  ~ReLU() override{};
  TensorVec forward(TensorVec &inputs) override;
  void backward() override;
 private:
  void backward_cpu();
  void backward_gpu();
  bool inplace_;
  SharedTensor result_;
 DISABLE_COPY_AND_ASSIGN(ReLU);
};
};//namespace nn
}

#endif //STENSOR_NN_RELU_LAYER_HPP_
