/**
* Copyright 2021 wss
* Created by wss on 11月,21, 2021
*/
#include "softmax_layer.hpp"
#include "common.hpp"

namespace stensor {
class SoftmaxTest : public ::testing::Test {};
TEST_F(SoftmaxTest, Forward) {
  int device_id = 0;
  nn::SharedTensor a(stensor::random({2, 2}, device_id));
  nn::TensorVec input;
  input.push_back(a);
  nn::SoftmaxLayer softmax_layer("mySoftmax", -1, device_id);

  nn::TensorVec output1 = softmax_layer.forward(input);
  nn::TensorVec output2 = softmax_layer.forward(input);

  output1[0]->to_cpu();
  output2[0]->to_cpu();
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(output1[0]->data_at(i), output2[0]->data_at(i));
  }

  a->to_cpu();
  std::cout << a << std::endl;
  std::cout << output1[0] << std::endl;
  std::cout << output2[0] << std::endl;
  std::cout << abi::__cxa_demangle(typeid(softmax_layer).name(), nullptr, nullptr, nullptr)<< std::endl;

  stensor::save(&softmax_layer,"/home/wss/CLionProjects/stensor/output/softmax_layer.pt3");
}

TEST_F(SoftmaxTest, load) {

  int device_id = 0;
  nn::SoftmaxLayer softmax_layer("mySoftmax", -1, device_id);


//  stensor::Config::set_random_seed(123);
  nn::SharedTensor a(stensor::ones({2, 2}, device_id));
  nn::TensorVec input;
  input.push_back(a);

  stensor::load(&softmax_layer,"/home/wss/CLionProjects/stensor/output/softmax_layer.pt3");

  nn::TensorVec output1 = softmax_layer.forward(input);
  nn::TensorVec output2 = softmax_layer.forward(input);

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

TEST_F(SoftmaxTest, Backward) {
  int device_id = 0;
  nn::SharedTensor a(stensor::random({2, 2}, device_id, true));
  nn::TensorVec input;
  input.push_back(a);
  nn::SoftmaxLayer softmax_layer("mySoftmax", -1, device_id);

  nn::TensorVec output1 = softmax_layer.forward(input);
  softmax_layer.backward();
  std::cout<<output1[0]->data_string();
  std::cout<<a->grad_string();
}
}