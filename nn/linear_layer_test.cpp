/**
* Copyright 2021 wss
* Created by wss on 11月,21, 2021
*/
#include "linear_layer.hpp"
#include "public/common.hpp"
#include "core/math_tesnsor.hpp"
#include <gtest/gtest.h>

namespace stensor {
class LinearLayerTest : public ::testing::Test {};
TEST_F(LinearLayerTest, Forward) {
  int device_id = 0;
  nn::SharedTensor a(stensor::random({2, 2}, device_id));
  nn::TensorVec input;
  input.push_back(a);
  nn::LinearLayer linear("myLinear", 2, 2, -1, device_id, true);

  nn::TensorVec output1 = linear.forward(input);
  Tensor b;
  b.copy_from(output1[0].get(), false, true);

  nn::TensorVec output2 = linear.forward(input);
  Tensor c;
  c.copy_from(output2[0].get(), false, true);

  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(b[i], c[i]);
  }

  std::cout << a << std::endl;
  std::cout << output1[0] << std::endl;
  std::cout << output2[0] << std::endl;
  std::cout << abi::__cxa_demangle(typeid(linear).name(), nullptr, nullptr, nullptr)<< std::endl;

  stensor::save(&linear,"/home/wss/CLionProjects/stensor/output/liner_layer.pt3");
}


TEST_F(LinearLayerTest, save) {
  int device_id = 0;
  stensor::Config::set_random_seed(123);
  nn::SharedTensor a(stensor::random({2, 2}, device_id));
  nn::TensorVec input;
  input.push_back(a);
  nn::LinearLayer linear("myLinear", 2, 2, -1, device_id, true);

  nn::TensorVec output1 = linear.forward(input);
  std::cout << output1[0] << std::endl;

  stensor::save(&linear,"/home/wss/CLionProjects/stensor/output/liner_layer.pt3");
}

TEST_F(LinearLayerTest, load) {

  int device_id = 0;
  nn::LinearLayer linear("myLinear", 2, 2, -1, device_id, true);

//  stensor::Config::set_random_seed(123);
  nn::SharedTensor a(stensor::ones({2, 2}, device_id));
  nn::TensorVec input;
  input.push_back(a);

  stensor::load(&linear,"/home/wss/CLionProjects/stensor/output/liner_layer.pt3");

  nn::TensorVec output1 = linear.forward(input);

  std::cout << a << std::endl;
  std::cout << output1[0] << std::endl;
}

TEST_F(LinearLayerTest, Backward) {
  int device_id = -1;
  nn::SharedTensor a(stensor::random({2, 2}, device_id));
  nn::TensorVec input;
  input.push_back(a);
  nn::LinearLayer linear("myLinear", 2, 2, -1, device_id, true);

  nn::TensorVec output1 = linear.forward(input);
  linear.backward();
}
}