/**
* Copyright 2021 wss
* Created by wss on 11月,15, 2021
*/
#include "math_base_cpu.hpp"
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
  Tensor t4(Tensor::ShapeType{3, 4});
  Tensor::Dtype *d4 = t4.mutable_cpu_data();
  // 1. set
  stensor::cpu_set(t1.size(), 1.0f, d1);
  stensor::cpu_set(t2.size(), 3.0f, d2);
  stensor::cpu_set(t3.size(), 0.0f, d3);
  for (int i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(d1[i], 1.0f);
  }

  stensor::cpu_scale(t1.size(), d1, 10.0f, d1);
  for (int i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(d1[i], 10.0f);
  }

  stensor::cpu_add_scalar(t1.size(), d1, 1.0f, d1);
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
  stensor::Config::set_random_seed(1);

  stensor::cpu_mul(t1.size(), d1, d2, d4);
  for (int i = 0; i < t3.size(); ++i) {
    EXPECT_EQ(d4[i], 33);
  }

  stensor::cpu_rng_uniform(t3.size(), 0.0f, 1.0f, d3);
  for (int i = 0; i < t3.size(); ++i) {
    std::cout << d3[i] << " ";
  }
  std::cout << std::endl;
}


TEST_F(MathTest, ActivateFUNC) {
  Tensor t1(Tensor::ShapeType{3, 4});
  Tensor::Dtype *d1 = t1.mutable_cpu_data();
  // 1. sigmoid
  stensor::cpu_set(t1.size(), 1.0f, d1);
  stensor::cpu_sigmoid(t1.size(), d1, d1);
  for (int i = 0; i < t1.size(); ++i) {
    EXPECT_LE(d1[i]-1.0 / (1.0 + exp(-1.0)), 1e-6);
  }

  // 2. sign
  stensor::cpu_set(t1.size(), 1.0f, d1);
  stensor::cpu_sign(t1.size(), d1, d1);
  for (int i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(d1[i], 1);
  }
  // 3. tanh
  stensor::cpu_set(t1.size(), 1.0f, d1);
  stensor::cpu_tanh(t1.size(), d1, d1);
  for (int i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(d1[i], std::tanh(1.0f));
  }

  // 4. relu
  stensor::cpu_set(t1.size(), -1.0f, d1);
  stensor::cpu_relu(t1.size(), d1, d1);
  for (int i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(d1[i], 0);
  }

}

}
