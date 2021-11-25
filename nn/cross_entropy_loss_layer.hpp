/**
* Copyright 2021 wss
* Created by wss on 11月,24, 2021
*/
#ifndef STENSOR_NN_CROSS_ENTROPY_LOSS_LAYER_HPP_
#define STENSOR_NN_CROSS_ENTROPY_LOSS_LAYER_HPP_
#include "module.hpp"
#include "math_tensor_backward.hpp"
#include "math_base_cuda.hpp"
#include "transpose.hpp"

namespace stensor {

namespace nn {
// softmax and
class CrossEntropyLossLayer : public Module {
 public:
  explicit CrossEntropyLossLayer(const std::string &name,
                                 int axis,
                                 int device);
  ~CrossEntropyLossLayer() {};
  TensorVec forward(TensorVec &inputs);
  inline void backward() {
    if (state_ == GPU)
      backward_gpu();
    else
      backward_cpu();
  }
  inline float get_loss() { return loss_; }
 private:
  inline void zero_temp_grad() {
    if (softmax_out_.get() && softmax_out_->require_grad())
      softmax_out_->zero_grad();
  }
  void backward_cpu();
  void backward_gpu();
  int axis_;
  float loss_;
  SharedTensor softmax_out_;
  SharedTensor one_hot_;
 DISABLE_COPY_AND_ASSIGN(CrossEntropyLossLayer);
};
}//namespace nn

}//namespace stensor
#endif //STENSOR_NN_CROSS_ENTROPY_LOSS_LAYER_HPP_
