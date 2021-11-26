/**
* Copyright 2021 wss
* Created by wss on 11月,21, 2021
*/
#include "cross_entropy_loss_layer.hpp"
#include "public/common.hpp"
#include "core/math_tesnsor.hpp"
#include <gtest/gtest.h>

namespace stensor {
class CrossEntropyTest : public ::testing::Test {};
TEST_F(CrossEntropyTest, Forward) {
  int device_id = 0;
  int batch_size = 2;
  int num_class = 2;

  nn::SharedTensor a(stensor::random({batch_size, num_class}, device_id, true));
  Tensor *gt = stensor::zeros({batch_size}, device_id);
  for (int i = 0; i < gt->size(); ++i) {
    (*gt)[i] = float(i);
  }

  std::cout << a;
  std::cout << gt;
  Tensor *sm = stensor::softmax(a.get(), -1);
  std::cout << sm << std::endl;

  nn::TensorVec input(2);
  input[0] = a;
  input[1].reset(gt);
  nn::CrossEntropyLossLayer cross_entropy_loss_layer("myCE", -1, device_id);

  cross_entropy_loss_layer.forward(input);
  std::cout << cross_entropy_loss_layer.get_loss() << std::endl;
  cross_entropy_loss_layer.forward(input);
  std::cout << cross_entropy_loss_layer.get_loss() << std::endl;

  Tensor *one_hot = stensor::one_hot(gt, num_class);
  Tensor *actual_grad = stensor::sub(sm, one_hot);
  cross_entropy_loss_layer.backward();
//  std::cout<<a->grad_string();
//  std::cout<<actual_grad->data_string();
  for (int i = 0; i < actual_grad->size(); ++i) {
    EXPECT_EQ(actual_grad->data_at(i), a->grad_at(i));
  }

  stensor::save(&cross_entropy_loss_layer,
                "/home/wss/CLionProjects/stensor/output/cross_entropy_loss_layer.pt3");
  delete sm;
  delete actual_grad;
  delete one_hot;
}

}