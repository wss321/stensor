/**
* Copyright 2021 wss
* Created by wss on 11月,21, 2021
*/
#include "linear_layer.hpp"
#include "common.hpp"

namespace stensor {
class LinearLayerTest : public ::testing::Test {};
TEST_F(LinearLayerTest, Forward) {
  int device_id = 0;
  nn::SharedTensor a(stensor::random({2, 2}, device_id));
  nn::TensorVec input;
  input.push_back(a);
  nn::LinearLayer *L = new nn::LinearLayer(2, 2, -1, device_id, true);

  nn::TensorVec output1 = L->forward(input);
  nn::TensorVec output2 = L->forward(input);

  output1[0]->to_cpu();
  output2[0]->to_cpu();
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(output1[0]->data_at(i), output2[0]->data_at(i));
  }

  a->to_cpu();
  std::cout << a << std::endl;
  std::cout << output1[0] << std::endl;
  std::cout << output2[0] << std::endl;
}
}