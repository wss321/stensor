/**
* Copyright 2021 wss
* Created by wss on 11月,21, 2021
*/
#ifndef STENSOR_NN_LINEAR_LAYER_HPP_
#define STENSOR_NN_LINEAR_LAYER_HPP_
#include "module.hpp"

namespace stensor {

namespace nn {
class LinearLayer : public Module {
 public:
  explicit LinearLayer(const std::string &name, int dim_in,
              int dim_out,
              int axis,
              int device,
              bool bias);
  ~LinearLayer(){};
  TensorVec forward(TensorVec &inputs);
  void backward();
 private:
//  TensorVec forward_cpu(TensorVec &inputs);
//  TensorVec forward_gpu(TensorVec &inputs);
  void backward_cpu();
  void backward_gpu();
  SharedTensor W_;
  SharedTensor b_;
  SharedTensor result_;
  int axis_;
  bool has_bias_;
 DISABLE_COPY_AND_ASSIGN(LinearLayer);
};
}//namespace nn

}//namespace stensor
#endif //STENSOR_NN_LINEAR_LAYER_HPP_
