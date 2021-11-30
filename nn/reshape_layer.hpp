/**
* Copyright 2021 wss
* Created by wss on 11月,30, 2021
*/
#ifndef STENSOR_NN_RESHAPE_LAYER_HPP_
#define STENSOR_NN_RESHAPE_LAYER_HPP_
#include <utility>

#include "module.hpp"

namespace stensor {

namespace nn {
class Reshape : public Module {
 public:
  explicit Reshape(std::string name, const std::vector<int> &new_shape) {
    name_ = std::move(name);
    type_ = "Reshape";
    out_shape_ = new_shape;
  };
  ~Reshape() override {};
  inline TensorVec forward(TensorVec &inputs) override {
    CHECK_EQ(inputs.size(), 1) << "Only support one input tensor now";
    state_ = inputs[0]->state();
    in_shape_ = inputs[0]->shape();
    inputs_ = inputs;
    for (int i = 0; i < out_shape_.size(); ++i) {
      if (out_shape_[i] == -1) out_shape_[i] = inputs[0]->shape(i);
    }
    inputs_[0]->reshape(out_shape_);
    this->zero_output_grad();
    outputs_ = inputs_;
    return outputs_;
  };
  void backward() override {
    outputs_[0]->reshape(in_shape_);
  }
 private:
  std::vector<int> out_shape_;
  std::vector<int> in_shape_;
 DISABLE_COPY_AND_ASSIGN(Reshape);
};
};//namespace nn
}// namespace stensor
#endif //STENSOR_NN_RESHAPE_LAYER_HPP_
