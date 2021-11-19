/**
* Copyright 2021 wss
* Created by wss on 11月,16, 2021
*/
#include "public/common.hpp"
#include <memory>
#include "tensor.hpp"
#include <vector>
#include "math_tesnsor.hpp"

namespace stensor {
class MathTensorTest : public ::testing::Test {};
TEST_F(MathTensorTest, Gennerator) {
  Tensor tensor(Tensor::ShapeType{3, 4});
  std::cout << stensor::zeros_like(&tensor) << std::endl;
  std::cout << stensor::ones_like(&tensor) << std::endl;
  std::cout << stensor::constants_like(&tensor, 5) << std::endl;
  std::cout << stensor::ones(std::vector<uint32_t>{5, 8}) << std::endl;
  std::cout << stensor::zeros(std::vector<uint32_t>{1, 3}) << std::endl;
  std::cout << stensor::random(std::vector<uint32_t>{1, 3}) << std::endl;
  std::cout << stensor::random_gaussian(std::vector<uint32_t>{1, 3}) << std::endl;
}

TEST_F(MathTensorTest, MathUnary) {
  Tensor::ShapeType shape1{3, 4};
  Tensor *a = stensor::random(shape1);
  Tensor *b = stensor::random(shape1);
  std::cout << "\ta:\n" << a;
  std::cout << "\tb:\n" << b;
  auto c = stensor::add(a, b);
  std::cout << "\ta+b:\n" << c;
//  for (int i = 0; i < a->size(); ++i) {
//    EXPECT_EQ(c->data_at(i), a->data_at(i) + b->data_at(i));
//  }
  auto d = stensor::exp(a);
  std::cout << "\texp:\n" << d;
//  for (int i = 0; i < a->size(); ++i) {
//    EXPECT_EQ(d->data_at(i), std::exp(a->data_at(i)));
//  }

  d = stensor::pow(a, 0.0);
  std::cout << "\tpow:\n" << d;
//  for (int i = 0; i < a->size(); ++i) {
//    EXPECT_EQ(d->data_at(i), std::pow(a->data_at(i), 0));
//  }
  d = stensor::mul(a, b);
  std::cout << "\tmul:\n" << d;
//  for (int i = 0; i < a->size(); ++i) {
//    EXPECT_EQ(d->data_at(i), a->data_at(i) * b->data_at(i));
//  }
  d = stensor::div(a, b);
  std::cout << "\tdiv:\n" << d;
}

TEST_F(MathTensorTest, MathRandomTest) {
  Tensor::ShapeType shape1{3, 4, 5};
  Tensor::ShapeType shape2{3, 1, 1};
  stensor::Config::set_random_seed(123);
  Tensor *a = stensor::random(shape1);
  stensor::Config::set_random_seed(123);
  Tensor *b = stensor::random(shape1);
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(a->data_at(i), b->data_at(i));
  }
}

TEST_F(MathTensorTest, MathBinary) {
  Tensor::ShapeType shape1{3, 4};
  Tensor *a = stensor::random(shape1);
  Tensor *b = stensor::random(shape1);
  std::cout << "\ta:\n" << a;
  std::cout << "\tb:\n" << b;
  auto d = stensor::mul(a, b);
  std::cout << "\tmul:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), a->data_at(i) * b->data_at(i));
  }
  d = stensor::div(a, b);
  std::cout << "\tdiv:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), a->data_at(i) / b->data_at(i));
  }
  d = stensor::add(a, b);
  std::cout << "\tadd:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), a->data_at(i) + b->data_at(i));
  }
  d = stensor::sub(a, b);
  std::cout << "\tsub:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), a->data_at(i) - b->data_at(i));
  }

}

TEST_F(MathTensorTest, MathCPUBroadCast) {
  Tensor::ShapeType shape1{3, 4, 5};
  Tensor::ShapeType shape2{3, 1, 1};
  Tensor::ShapeType shape3{3, 4, 5};
//  Tensor *a = stensor::random(shape1);
//  Tensor *b = stensor::random(shape2);
  Tensor *a = stensor::ones(shape1);
  Tensor *b = stensor::constants(shape2, 2);

  std::cout << "\ta:\n" << a;
  std::cout << "\tb:\n" << b;
  auto c = stensor::add(a, b);
  std::cout << "\ta+b:\n" << c;
  EXPECT_EQ(c->shape(), shape3);
  c = stensor::sub(a, b);
  std::cout << "\ta-b:\n" << c;
  EXPECT_EQ(c->shape(), shape3);
  c = stensor::mul(a, b);
  std::cout << "\ta*b:\n" << c;
  EXPECT_EQ(c->shape(), shape3);
  c = stensor::div(a, b);
  std::cout << "\ta/b:\n" << c;
  EXPECT_EQ(c->shape(), shape3);
}

TEST_F(MathTensorTest, MathGPUBroadCast) {
  Tensor::ShapeType shape1{3, 4, 5};
  Tensor::ShapeType shape2{4, 1};
  Tensor::ShapeType shape3{3, 4, 5};
//  Tensor *a = stensor::random(shape1);
//  Tensor *b = stensor::random(shape2);
  Tensor *a = stensor::ones(shape1);
  Tensor *b = stensor::constants(shape2, 2.5);
  a->to_gpu();
  b->to_gpu();
  EXPECT_EQ(a->device(), 0);
  std::cout << "\ta:\n" << a;
  std::cout << "\tb:\n" << b;
  Tensor *c = stensor::add(a, b);
  std::cout << "\tgpu:c=a+b:\n" << c;
  Tensor *d = stensor::sub(a, b);
  std::cout << "\tgpu:c=a-b:\n" << d;
  Tensor *e = stensor::mul(a, b);
  std::cout << "\tgpu:c=a*b:\n" << e;
  Tensor *f = stensor::div(a, b);
  std::cout << "\tgpu:c=a/b:\n" << f;
  delete a;
  delete b;
  delete c;
  delete d;
  delete e;
  delete f;

}

TEST_F(MathTensorTest, MathGPUBroadCastSpeed) {
  Tensor::ShapeType shape1{128, 3, 224, 224};
  Tensor::ShapeType shape2{3, 224, 224};
  Tensor *a = stensor::random(shape1);
  Tensor *b = stensor::random(shape2);
  long long start = systemtime_ms();
  Tensor *c1 = stensor::add(a, b);
  Tensor *d1 = stensor::sub(a, b);
  Tensor *e1 = stensor::mul(a, b);
  Tensor *f1 = stensor::div(a, b);
  LOG(INFO) << "CPU broadcast operation time:" << systemtime_ms() - start << "ms";

  a->to_gpu();
  b->to_gpu();
  EXPECT_EQ(a->device(), 0);
  start = systemtime_ms();
  Tensor *c = stensor::add(a, b);
  Tensor *d = stensor::sub(a, b);
  Tensor *e = stensor::mul(a, b);
  Tensor *f = stensor::div(a, b);
  LOG(INFO) << "GPU broadcast operation time:" << systemtime_ms() - start << "ms";

  delete a;
  delete b;
  delete c;
  delete d;
  delete e;
  delete f;
  delete c1;
  delete d1;
  delete e1;
  delete f1;

}


TEST_F(MathTensorTest, ActivateFunc) {
  Tensor::ShapeType shape1{3, 4};
  Tensor *a = stensor::random(shape1, -1.0f, 1.0f);
  std::cout << "\ta:\n" << a;
  auto d = stensor::sigmoid(a);
  std::cout << "\tsigmoid:\n" << d;
  for (int index = 0; index < a->size(); ++index) {
    float x=a->data_at(index);
    float y=d->data_at(index);
    EXPECT_EQ(1.0f/(1.0f + std::exp(-x)), y);
  }
  delete d;

  d = stensor::tanh(a);
  std::cout << "\ttanh:\n" << d;
  for (int index = 0; index < a->size(); ++index) {
    float x=a->data_at(index);
    float y=d->data_at(index);
    EXPECT_EQ(std::tanh(x), y);
  }
  delete d;
  d = stensor::relu(a);
  std::cout << "\trelu:\n" << d;
  for (int index = 0; index < a->size(); ++index) {
    float x=a->data_at(index);
    float y=d->data_at(index);
    EXPECT_EQ(std::max(x, 0.0f), y);
  }
  delete d;
//  d = stensor::tanh(a);
//  std::cout << "\telu:\n" << d;
//  for (int index = 0; index < a->size(); ++index) {
//    float x=a->data_at(index);
//    float y=d->data_at(index);
//    EXPECT_EQ(std::tanh(x), y);
//  }
}
}