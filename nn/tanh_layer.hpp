/**
* Copyright 2021 wss
* Created by wss on 11月,26, 2021
*/
#ifndef STENSOR_NN_TANH_LAYER_HPP_
#define STENSOR_NN_TANH_LAYER_HPP_
#include "module.hpp"

namespace stensor {

namespace nn {
class TanH : public Module {
 public:
  explicit TanH(std::string name, bool inplace = false) :
      inplace_(inplace) {
    name_ = name;
    type_ = "TanH";
  };
  ~TanH() {};
  TensorVec forward(TensorVec &inputs);
  void backward();
 private:
  void backward_cpu();
  void backward_gpu();
  bool inplace_;
  SharedTensor result_;
 DISABLE_COPY_AND_ASSIGN(TanH);
};
};//namespace nn
}// namespace stensor
#endif //STENSOR_NN_TANH_LAYER_HPP_
