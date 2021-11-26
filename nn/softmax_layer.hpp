/**
* Copyright 2021 wss
* Created by wss on 11月,24, 2021
*/
#ifndef STENSOR_NN_SOFTMAX_LAYER_HPP_
#define STENSOR_NN_SOFTMAX_LAYER_HPP_
#include "module.hpp"

namespace stensor {

namespace nn {
class Softmax : public Module {
 public:
  explicit Softmax(const std::string &name,
                   int axis,
                   int device);
  ~Softmax() {};
  TensorVec forward(TensorVec &inputs);
  inline void backward() {
    if (state_ == GPU)
      backward_gpu();
    else
      backward_cpu();
  }
 private:
  void backward_cpu();
  void backward_gpu();
  int axis_;
 DISABLE_COPY_AND_ASSIGN(Softmax);
};
}//namespace nn

}//namespace stensor

#endif //STENSOR_NN_SOFTMAX_LAYER_HPP_
