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
class CPUMathTest : public ::testing::Test {};

TEST_F(CPUMathTest, SelfOp) {
  Tensor t1(Tensor::ShapeType{3, 4});
  Tensor::Dtype *d1 = t1.data();

  Tensor t2(Tensor::ShapeType{3, 4});
  Tensor::Dtype *d2 = t2.data();

  Tensor t3(Tensor::ShapeType{3, 3});
  Tensor::Dtype *d3 = t3.data();
  Tensor t4(Tensor::ShapeType{3, 4});
  Tensor::Dtype *d4 = t4.data();
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


TEST_F(CPUMathTest, ActivateFUNC) {
  Tensor t1(Tensor::ShapeType{3, 4});
  Tensor::Dtype *d1 = t1.data();
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


TEST_F(CPUMathTest, CompTest) {
  int size1 = 4 * 10;
  int size2 = 4 * 10;
  int size3 = 4 * 10;

  SynMem A(size1 * sizeof(float));
  SynMem B(size2 * sizeof(float));
  SynMem C(size3 * sizeof(float));

  float *g1 = (float *) A.cpu_data();
  float *g2 = (float *) B.cpu_data();
  float *g3 = (float *) C.cpu_data();

  stensor::cpu_rng_uniform<float>(size1, -1.0, 1.0, g1);
  stensor::cpu_rng_uniform<float>(size2, 0.0, 1.0, g2);

  bool iseq = cpu_equal(size1,  g1, g2);
  EXPECT_EQ(false, iseq);

  iseq = cpu_equal(size1,  g1, g1);
  EXPECT_EQ(true, iseq);

  cpu_maximum(size1, g1, g2, g3);
  for (int i = 0; i < size1; ++i) {
    EXPECT_EQ(std::max(g1[i], g2[i]), g3[i]);
  }
  cpu_minimum(size1, g1, g2, g3);
  for (int i = 0; i < size1; ++i) {
    EXPECT_EQ(std::min(g1[i], g2[i]), g3[i]);
  }



}

}
