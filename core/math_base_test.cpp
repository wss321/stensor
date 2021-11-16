/**
* Copyright 2021 wss
* Created by wss on 11月,15, 2021
*/
#include "math_base.hpp"
#include "public/common.hpp"
#include <memory>
#include "tensor.hpp"
#include <vector>

namespace stensor {
class MathTest : public ::testing::Test {};

TEST_F(MathTest, SelfOp) {
  Tensor t1(Tensor::ShapeType{3, 4});
  Tensor::Dtype *d1 = t1.mutable_cpu_data();

  Tensor t2(Tensor::ShapeType{3, 4});
  Tensor::Dtype *d2 = t2.mutable_cpu_data();

  Tensor t3(Tensor::ShapeType{3, 3});
  Tensor::Dtype *d3 = t3.mutable_cpu_data();
  // 1. set
  stensor::set(t1.size(), 1.0f, d1);
  stensor::set(t2.size(), 3.0f, d2);
  stensor::set(t3.size(), 0.0f, d3);
  for (int i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(d1[i], 1.0f);
  }

  stensor::scale(t1.size(), d1, 10.0f, d1);
  for (int i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(d1[i], 10.0f);
  }

  stensor::add(t1.size(), d1, 1.0f, d1);
  for (int i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(d1[i], 11.0f);
  }

  stensor::cpu_gemm(CblasNoTrans, CblasTrans,
                    3, 3, 4,
                    1.0f, d1, d2,
                    1.0f, d3);
  for (int i = 0; i < t3.size(); ++i) {
    EXPECT_EQ(d3[i], 132);
  }

  stensor::mul(t1.size(), d1, d2, d3);
  for (int i = 0; i < t3.size(); ++i) {
    EXPECT_EQ(d3[i], 33);
  }
  stensor::rng_gaussian(t3.size(), 0.0f, 1.0f, d3);
  for (int i = 0; i < t3.size(); ++i) {
    std::cout << d3[i] << " ";
  }
  std::cout << std::endl;
}
}
