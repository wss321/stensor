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
class TensorTest : public ::testing::Test {};
TEST_F(TensorTest, Gennerator) {
  Tensor tensor(Tensor::ShapeType{3, 4});
  std::cout << stensor::zeros_like(&tensor) << std::endl;
  std::cout << stensor::ones_like(&tensor) << std::endl;
  std::cout << stensor::constants_like(&tensor, 5) << std::endl;
  std::cout << stensor::ones(std::vector<uint32_t>{5, 8}) << std::endl;
  std::cout << stensor::zeros(std::vector<uint32_t>{1, 3}) << std::endl;
  std::cout << stensor::random(std::vector<uint32_t>{1, 3}) << std::endl;
  std::cout << stensor::random_gaussian(std::vector<uint32_t>{1, 3}) << std::endl;
}

TEST_F(TensorTest, MathUnary) {
  Tensor::ShapeType shape1{3, 4};
  Tensor *a = stensor::random(shape1);
  Tensor *b = stensor::random(shape1);
  std::cout << "\ta:\n" << a;
  std::cout << "\tb:\n" << b;
  auto c = stensor::add(a, b);
  std::cout << "\ta+b:\n" << c;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(c->data_at(i), a->data_at(i) + b->data_at(i));
  }
  auto d = stensor::exp(a);
  std::cout << "\texp:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), std::exp(a->data_at(i)));
  }

  d = stensor::pow(a, 0.0);
  std::cout << "\tpow:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), std::pow(a->data_at(i), 0));
  }
  d = stensor::mul(a, b);
  std::cout << "\tmul:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), a->data_at(i) * b->data_at(i));
  }
  d = stensor::div(a, b);
  std::cout << "\tdiv:\n" << d;
}

TEST_F(TensorTest, MathBinary) {
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

TEST_F(TensorTest, MathBroadCast) {
  Tensor::ShapeType shape1{3, 4, 5};
  Tensor::ShapeType shape2{3, 1, 1};
  Tensor::ShapeType shape3{3, 4, 5};
  Tensor *a = stensor::random(shape1);
  Tensor *b = stensor::random(shape2);
//  Tensor *a = stensor::ones(shape1);
//  Tensor *b = stensor::constants(shape2, 2);

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

}