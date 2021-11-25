/**
* Copyright 2021 wss
* Created by wss on 11月,24, 2021
*/
#ifndef STENSOR_NN_SOFTMAX_LAYER_HPP_
#define STENSOR_NN_SOFTMAX_LAYER_HPP_
#include "module.hpp"
#include "math_tensor_backward.hpp"
#include "math_base_cuda.hpp"
namespace stensor {

namespace nn {
class SoftmaxLayer : public Module {
 public:
  explicit SoftmaxLayer(const std::string &name,
                        int axis,
                        int device);
  ~SoftmaxLayer() {};
  std::vector<Tensor*> forward(std::vector<Tensor*> &inputs);
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
 DISABLE_COPY_AND_ASSIGN(SoftmaxLayer);
};
}//namespace nn

}//namespace stensor

#endif //STENSOR_NN_SOFTMAX_LAYER_HPP_
