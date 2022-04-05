/**
* Copyright 2021 wss
* Created by wss on 11月,16, 2021
*/
#include "math_base_cuda.hpp"
#include "public/synmem.hpp"
#include "public/common.hpp"
#include "public/memory_op.hpp"
#include "math_base_cpu.hpp"
#include "core/tensor.hpp"
#include <gtest/gtest.h>

namespace stensor {
class GPUMathTest : public ::testing::Test {};

TEST_F(GPUMathTest, MemTest) {
  uint32_t size = 1024; //byte
  SynMem data(size);
  LOG(INFO) << data.device();

  data.alloc_gpu();
  void *gpu_m = data.gpu_data();
  EXPECT_EQ(data.device(), 0);

  void *cpu_m = data.cpu_data();
  EXPECT_EQ(data.device(), 0);

  cpu_memset(size, 1, cpu_m);
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(((char *) cpu_m)[i], 1);
  }

  data.copy_cpu_to_gpu();
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(((char *) gpu_m)[i], 1);
  }

  EXPECT_EQ(data.device(), 0);

}

TEST_F(GPUMathTest, MMTest) {
  int size1 = 4 * 10;
  int size2 = 4 * 10;
  int size3 = 4 * 10;

  SynMem A(size1 * sizeof(float));
  SynMem B(size2 * sizeof(float));
  SynMem C(size3 * sizeof(float));
//  stensor::gpu_memset(2 * sizeof(float), 1, A.mutable_gpu_data());
//  for (int i = 0; i < 2 * sizeof(float); ++i) {
//    EXPECT_EQ(((char *) A.gpu_data())[i], 1);
//  }
  A.alloc_gpu();
  B.alloc_gpu();
  C.alloc_gpu();
  float *g1 = (float *) A.gpu_data();
  float *g2 = (float *) B.gpu_data();
  float *g3 = (float *) C.gpu_data();

  const float *g1c = (const float *) A.cpu_data();
  const float *g2c = (const float *) B.cpu_data();
  const float *g3c = (const float *) C.cpu_data();

  stensor::gpu_rng_uniform<float>(size1, -1.0, 1.0, g1);
  stensor::gpu_rng_uniform<float>(size2, 0.0, 1.0, g2);

  A.copy_gpu_to_cpu();
  B.copy_gpu_to_cpu();
  C.copy_gpu_to_cpu();
  for (int i = 0; i < size1; ++i) {
    EXPECT_EQ(g1c[i], g1[i]);
//    LOG(INFO) << std::abs(g1c[i]) << " " << g3c[i];
  }

  stensor::gpu_abs<float>(size1, g1, g3);
  C.copy_gpu_to_cpu();
  for (int i = 0; i < size1; ++i) {
    EXPECT_EQ(std::abs(g1[i]), g3c[i]);
//    LOG(INFO) << std::abs(g1c[i]) << " " << g3c[i];
  }

  stensor::gpu_div<float>(size1, g1, g2, g3);
  C.copy_gpu_to_cpu();
  for (int i = 0; i < size1; ++i) {
    EXPECT_EQ(g1c[i] / g2c[i], g3c[i]);
//    LOG(INFO) << g1c[i] / g2c[i] << " " << g3c[i];
  }

  stensor::gpu_mul<float>(size1, g1, g2, g3);
  C.copy_gpu_to_cpu();
  for (int i = 0; i < size1; ++i) {
    EXPECT_EQ(g1c[i] * g2c[i], g3c[i]);
  }

  float ans = 0;
  for (int i = 0; i < size1; ++i) {
    ans += g1c[i] * g2c[i];
  }

  stensor::gpu_dot<float>(size1, g1, g2, g3);
  C.copy_gpu_to_cpu();
  EXPECT_LE(std::abs(ans - *g3c), 1e-5);

  stensor::gpu_sub<float>(size1, g1, g2, g3);
  C.copy_gpu_to_cpu();
  for (int i = 0; i < size1; ++i) {
    EXPECT_EQ(g1c[i] - g2c[i], g3c[i]);
  }

  stensor::gpu_add_scalar<float>(size1, g1, 3, g1);
  stensor::gpu_set<float>(size1, 3.0, g1);
  C.copy_gpu_to_cpu();
  for (int i = 0; i < size1; ++i) {
    EXPECT_EQ(g1c[i] - g2c[i], g3c[i]);
  }
}

TEST_F(GPUMathTest, SpeedTest) {
  int N = 10000;
  int size1 = N * N;
  int size2 = N * N;
  int size3 = N * N;

  SynMem A(size1 * sizeof(float));
  SynMem B(size2 * sizeof(float));
  SynMem C(size3 * sizeof(float));
  A.alloc_gpu();
  B.alloc_gpu();
  C.alloc_gpu();
  float *g1 = (float *) A.gpu_data();
  float *g2 = (float *) B.gpu_data();
  float *g3 = (float *) C.gpu_data();

  float *c1 = (float *) A.cpu_data();
  float *c2 = (float *) B.cpu_data();
  float *c3 = (float *) C.cpu_data();
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
  Tensor t1(Tensor::ShapeType{5, 6}, 0);
  Tensor::Dtype *gpu_data = t1.data();
  // 1. sigmoid
  stensor::gpu_set(t1.size(), 1.0f, gpu_data);
  stensor::gpu_sigmoid(t1.size(), gpu_data, gpu_data);
  t1.to_cpu();
  Tensor::Dtype *cpu_data = t1.data();
  for (int i = 0; i < t1.size(); ++i) {
    EXPECT_LE(t1[i] - 1.0f / (1.0f + exp(-1.0f)), 1e-6);
  }

  // 2. sign
  t1.to_gpu();
  gpu_data = t1.data();
  stensor::gpu_set(t1.size(), 1.0f, gpu_data);
  stensor::gpu_sign(t1.size(), gpu_data, gpu_data);
  t1.to_cpu();
  for (int i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(t1[i], 1.0f);
  }
  // 3. tanh
  t1.to_gpu();
  gpu_data = t1.data();
  stensor::gpu_set(t1.size(), 1.0f, gpu_data);
  stensor::gpu_tanh(t1.size(), gpu_data, gpu_data);
  t1.to_cpu();
  for (int i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(t1[i], std::tanh(1.0f));
  }

  // 4. relu
  t1.to_gpu();
  gpu_data = t1.data();
  stensor::gpu_set(t1.size(), -1.0f, gpu_data);
  stensor::gpu_relu(t1.size(), gpu_data, gpu_data);
  t1.to_cpu();
  cpu_data = t1.data();
  for (int i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(t1[i], 0.0f);
  }

}

TEST_F(GPUMathTest, CompTest) {
  int size1 = 4 * 10;
  int size2 = 4 * 10;
  int size3 = 4 * 10;

  SynMem A(size1 * sizeof(float));
  SynMem B(size2 * sizeof(float));
  SynMem C(size3 * sizeof(float));
  A.alloc_gpu();
  B.alloc_gpu();
  C.alloc_gpu();

  float *g1 = (float *) A.gpu_data();
  float *g2 = (float *) B.gpu_data();
  float *g3 = (float *) C.gpu_data();

  const float *g1c = (const float *) A.cpu_data();
  const float *g2c = (const float *) B.cpu_data();
  const float *g3c = (const float *) C.cpu_data();

  stensor::gpu_rng_uniform<float>(size1, -1.0, 1.0, g1);
  stensor::gpu_rng_uniform<float>(size2, 0.0, 1.0, g2);

  A.copy_gpu_to_cpu();
  B.copy_gpu_to_cpu();
  C.copy_gpu_to_cpu();

  bool iseq = gpu_equal(size1, g1, g2);
  C.copy_gpu_to_cpu();
  EXPECT_EQ(false, iseq);

  iseq = gpu_equal(size1, g1, g1);
  C.copy_gpu_to_cpu();
  EXPECT_EQ(true, iseq);

  gpu_maximum(size1, g1, g2, g3);
  C.copy_gpu_to_cpu();
  for (int i = 0; i < size1; ++i) {
    EXPECT_EQ(std::max(g1c[i], g2c[i]), g3c[i]);
  }

  gpu_minimum(size1, g1, g2, g3);
  C.copy_gpu_to_cpu();
  for (int i = 0; i < size1; ++i) {
    EXPECT_EQ(std::min(g1c[i], g2c[i]), g3c[i]);
  }

}

TEST_F(GPUMathTest, ReduceTest) {
  int M = 4, D = 10, N = 2;
  int size1 = M*D*N;
  int size2 = M*N;


  SynMem A(size1 * sizeof(float));
  SynMem B(size2 * sizeof(float));
  A.alloc_gpu();
  B.alloc_gpu();

  float *g1 = (float *) A.gpu_data();
  float *g2 = (float *) B.gpu_data();
  for (int i = 0; i < size1; ++i) {
    g1[i] = i%D;
    std::cout<<g1[i]<<", ";
  }
  std::cout<<std::endl;

  stensor::gpu_reduce_mean(M, D, N, g1, 0.0f, g2);
  for (int i = 0; i < M*N; ++i) {
    std::cout<<g2[i]<<", ";
  }
  std::cout<<std::endl;

  stensor::gpu_reduce_var(M, D, N, g1, 0.0f, g2);
  for (int i = 0; i < M*N; ++i) {
    std::cout<<g2[i]<<", ";
  }
  std::cout<<std::endl;

  stensor::gpu_reduce_std(M, D, N, g1, 0.0f, g2);
  for (int i = 0; i < M*N; ++i) {
    std::cout<<g2[i]<<", ";
  }
  std::cout<<std::endl;

//  const float *g1c = (const float *) A.cpu_data();
//  const float *g2c = (const float *) B.cpu_data();
//
//  stensor::gpu_rng_uniform<float>(size1, -1.0, 1.0, g1);
//  stensor::gpu_rng_uniform<float>(size2, 0.0, 1.0, g2);
//
//  A.copy_gpu_to_cpu();
//  B.copy_gpu_to_cpu();


}
}
