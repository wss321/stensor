/**
* Copyright 2021 wss
* Created by wss on 11月,21, 2021
*/
#ifndef STENSOR_NN_LINEAR_LAYER_HPP_
#define STENSOR_NN_LINEAR_LAYER_HPP_
#include "module.hpp"
#include "math_tensor_backward.hpp"

namespace stensor {

namespace nn {
class LinearLayer : public Module {
 public:
  LinearLayer(int dim_in, int dim_out, int axis=-1, int device=-1, bool bias=true);
  TensorVec forward(TensorVec &inputs);
  void backward();
 private:
  SharedTensor W_;
  SharedTensor b_;
  int axis_;
  bool has_bias_;
  TensorVec bottom_;
  TensorVec top_;

};
}//namespace nn

}//namespace stensor
#endif //STENSOR_NN_LINEAR_LAYER_HPP_
