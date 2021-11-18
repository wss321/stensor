/**
* Copyright 2021 wss
* Created by wss on 11月,16, 2021
*/
#include "math_base_cuda.hpp"
#include "public/synmem.hpp"
#include "common.hpp"
#include "memory_op.hpp"
#include "math_base_cpu.hpp"
#include "tensor.hpp"

namespace stensor {
class GPUMathTest : public ::testing::Test {};

TEST_F(GPUMathTest, MemTest) {
  uint32_t size = 1024; //byte
  SynMem data(size);
  LOG(INFO) << data.device();

  void *gpu_m = data.mutable_gpu_data();
  EXPECT_EQ(data.state(), SynMem::AT_GPU);

  void *cpu_m = data.mutable_cpu_data();
  EXPECT_EQ(data.state(), SynMem::AT_CPU);

  cpu_memset(size, 1, cpu_m);
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(((char *) cpu_m)[i], 1);
  }

  data.to_gpu();
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(((char *) gpu_m)[i], 1);
  }

  EXPECT_EQ(data.device(), 0);

}

TEST_F(GPUMathTest, MMTest) {
  int size1 = 2 * 10;
  int size2 = 2 * 10;
  int size3 = 2 * 10;

  SynMem A(size1 * sizeof(float));
  SynMem B(size2 * sizeof(float));
  SynMem C(size3 * sizeof(float));
//  stensor::gpu_memset(2 * sizeof(float), 1, A.mutable_gpu_data());
//  for (int i = 0; i < 2 * sizeof(float); ++i) {
//    EXPECT_EQ(((char *) A.gpu_data())[i], 1);
//  }
  float *g1 = (float *) A.mutable_gpu_data();
  float *g2 = (float *) B.mutable_gpu_data();
  float *g3 = (float *) C.mutable_gpu_data();

  const float *g1c = (const float *) A.gpu_data();
  const float *g2c = (const float *) B.gpu_data();
  const float *g3c = (const float *) C.gpu_data();

  stensor::gpu_rng_uniform<float>(size1, -1.0, 1.0, g1);
  stensor::gpu_rng_uniform<float>(size2, 0.0, 1.0, g2);
  stensor::gpu_abs<float>(size1, g1c, g3);
  for (int i = 0; i < size1; ++i) {
    EXPECT_EQ(std::abs(g1c[i]), g3c[i]);
  }

  stensor::gpu_div<float>(size1, g1c, g2c, g3);
  for (int i = 0; i < size1; ++i) {
    EXPECT_EQ(g1c[i] / g2c[i], g3c[i]);
//    LOG(INFO) << g1c[i] / g2c[i] << " " << g3c[i];
  }
  stensor::gpu_mul<float>(size1, g1c, g2c, g3);
  for (int i = 0; i < size1; ++i) {
    EXPECT_EQ(g1c[i] * g2c[i], g3c[i]);
//    LOG(INFO) << g1c[i] / g2c[i] << " " << g3c[i];
  }
  stensor::gpu_dot<float>(size1, g1c, g2c, g3);
  for (int i = 0; i < size1; ++i) {
    EXPECT_EQ(g1c[i] * g2c[i], g3c[i]);
//    LOG(INFO) << g1c[i] * g2c[i] << " " << g3c[i];
  }

  stensor::gpu_sub<float>(size1, g1c, g2c, g3);
  for (int i = 0; i < size1; ++i) {
    EXPECT_EQ(g1c[i] - g2c[i], g3c[i]);
//    LOG(INFO) << g1c[i] - g2c[i] << " " << g3c[i];
  }
  stensor::gpu_add_scalar<float>(size1, g1c, 3, g1);
  stensor::gpu_set<float>(size1, 3.0, g1);
  for (int i = 0; i < size1; ++i) {
    EXPECT_EQ(g1c[i] - g2c[i], g3c[i]);
//    LOG(INFO) << g1c[i] - g2c[i] << " " << g3c[i];
  }
}

TEST_F(GPUMathTest, SpeedTest) {
  int N = 2000;
  int size1 = N * N;
  int size2 = N * N;
  int size3 = N * N;

  SynMem A(size1 * sizeof(float));
  SynMem B(size2 * sizeof(float));
  SynMem C(size3 * sizeof(float));

  float *g1 = (float *) A.mutable_gpu_data();
  float *g2 = (float *) B.mutable_gpu_data();
  float *g3 = (float *) C.mutable_gpu_data();

  float *c1 = (float *) A.mutable_cpu_data();
  float *c2 = (float *) B.mutable_cpu_data();
  float *c3 = (float *) C.mutable_cpu_data();
  long long start = systemtime_ms();
  stensor::gpu_rng_uniform<float>(size1, -1.0, 1.0, g1);
  stensor::gpu_rng_uniform<float>(size2, -1.0, 1.0, g2);
  LOG(INFO) << "GPU RNG time:" << systemtime_ms() - start << "ms";

  start = systemtime_ms();
  stensor::cpu_rng_uniform<float>(size1, -1, 1, c1);
  stensor::cpu_rng_uniform<float>(size2, -1, 1, c2);
  LOG(INFO) << "CPU RNG time:" << systemtime_ms() - start << "ms";

  start = systemtime_ms();
  stensor::gpu_dot(size1, g1, g2, g3);
  LOG(INFO) << "GPU dot time:" << systemtime_ms() - start << "ms";

  start = systemtime_ms();
  stensor::cpu_dot(size1, c1, c2);
  LOG(INFO) << "CPU dot time:" << systemtime_ms() - start << "ms";

  // add
  start = systemtime_ms();
  stensor::gpu_add(size1, g1, g2, g3);
  LOG(INFO) << "GPU add time:" << systemtime_ms() - start << "ms";

  start = systemtime_ms();
  stensor::cpu_add(size1, c1, c2, c3);
  LOG(INFO) << "CPU add time:" << systemtime_ms() - start << "ms";

  start = systemtime_ms();
  stensor::gpu_gemm<float>(CblasNoTrans, CblasNoTrans,
                           N, N, N,
                           1, g1, g2, 0, g3);
  LOG(INFO) << "GPU matmul time:" << systemtime_ms() - start << "ms";

  start = systemtime_ms();
  stensor::cpu_gemm<float>(CblasNoTrans, CblasNoTrans,
                           N, N, N,
                           1, c1, c2, 0, c3);
  LOG(INFO) << "CPU matmul time:" << systemtime_ms() - start << "ms";
//  for (int i = 0; i < 100; ++i) {
//    EXPECT_EQ(c3[i], g3[i]);
//  }

}


TEST_F(GPUMathTest, ActivateFUNC) {
  Tensor t1(Tensor::ShapeType{3, 40});
  Tensor::Dtype *d1 = t1.mutable_gpu_data();
  // 1. sigmoid
  stensor::gpu_set(t1.size(), 1.0f, d1);
  stensor::gpu_sigmoid(t1.size(), d1, d1);
  for (int i = 0; i < t1.size(); ++i) {
    EXPECT_LE(d1[i]-1.0f / (1.0f + exp(-1.0f)), 1e-6);
  }

  // 2. sign
  stensor::gpu_set(t1.size(), 1.0f, d1);
  stensor::gpu_sign(t1.size(), d1, d1);
  for (int i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(d1[i], 1.0f);
  }
  // 3. tanh
  stensor::gpu_set(t1.size(), 1.0f, d1);
  stensor::gpu_tanh(t1.size(), d1, d1);
  for (int i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(d1[i], std::tanh(1.0f));
  }

  // 4. relu
  stensor::gpu_set(t1.size(), -1.0f, d1);
  stensor::gpu_relu(t1.size(), d1, d1);
  for (int i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(d1[i], 0.0f);
  }

}

}
