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
  std::vector<Tensor*> input;
  input.push_back(a.get());
  nn::LinearLayer linear("myLinear", 2, 2, -1, device_id, true);

  std::vector<Tensor*> output1 = linear.forward(input);
  std::vector<Tensor*> output2 = linear.forward(input);

  output1[0]->to_cpu();
  output2[0]->to_cpu();
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(output1[0]->data_at(i), output2[0]->data_at(i));
  }

  a->to_cpu();
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
  std::vector<Tensor*> input;
  input.push_back(a.get());
  nn::LinearLayer linear("myLinear", 2, 2, -1, device_id, true);

  std::vector<Tensor*> output1 = linear.forward(input);
  std::vector<Tensor*> output2 = linear.forward(input);

  output1[0]->to_cpu();
  output2[0]->to_cpu();
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(output1[0]->data_at(i), output2[0]->data_at(i));
  }

  a->to_cpu();
  std::cout << a << std::endl;
  std::cout << output1[0] << std::endl;
  std::cout << output2[0] << std::endl;
  std::cout << abi::__cxa_demangle(typeid(linear).name(), nullptr, nullptr, nullptr)<< std::endl;

  stensor::save(&linear,"/home/wss/CLionProjects/stensor/output/liner_layer.pt3");
}

TEST_F(LinearLayerTest, load) {

  int device_id = 0;
  nn::LinearLayer linear("myLinear", 2, 2, -1, device_id, true);

//  stensor::Config::set_random_seed(123);
  nn::SharedTensor a(stensor::ones({2, 2}, device_id));
  std::vector<Tensor*> input;
  input.push_back(a.get());

  stensor::load(&linear,"/home/wss/CLionProjects/stensor/output/liner_layer.pt3");

  std::vector<Tensor*> output1 = linear.forward(input);
  std::vector<Tensor*> output2 = linear.forward(input);

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

TEST_F(LinearLayerTest, Backward) {
  int device_id = -1;
  nn::SharedTensor a(stensor::random({2, 2}, device_id));
  std::vector<Tensor*> input;
  input.push_back(a.get());
  nn::LinearLayer linear("myLinear", 2, 2, -1, device_id, true);

  std::vector<Tensor*> output1 = linear.forward(input);
  linear.backward();
}
}