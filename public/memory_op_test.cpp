/**
* Copyright 2021 wss
* Created by wss on 11月,16, 2021
*/
#include "common.hpp"
#include "memory_op.hpp"
#include <gtest/gtest.h>

namespace stensor {

class MemOpTest : public ::testing::Test {};

TEST_F(MemOpTest, MallocTest) {
  uint32_t Gsize = 1024 * 1024 * 1024; //byte
  void *cpu_m;
  MallocCPU(&cpu_m, Gsize);
  void *gpu_m;
  MallocGPU(&gpu_m, Gsize);

  FreeCPU(cpu_m);
  FreeGPU(gpu_m);
  std::cout << "Done.";

}
TEST_F(MemOpTest, MemSetTest) {
  uint32_t Gsize = 1024; //byte
  void *cpu_m;
  MallocCPU(&cpu_m, Gsize);
  void *gpu_m;
  MallocGPU(&gpu_m, Gsize);
  cpu_memset(Gsize, 1, cpu_m);
  for (int i = 0; i < Gsize; ++i) {
    EXPECT_EQ(((char *) cpu_m)[i], 1);
  }

  gpu_memset(Gsize, 1, gpu_m);
  for (int i = 0; i < Gsize; ++i) {
    EXPECT_EQ(((char *) gpu_m)[i], 1);
  }
  FreeCPU(cpu_m);
  FreeGPU(gpu_m);

}

TEST_F(MemOpTest, MemCopyTest) {
  uint32_t Gsize = 1024; //byte
  void *cpu_m;
  MallocCPU(&cpu_m, Gsize);
  void *gpu_m;
  MallocGPU(&gpu_m, Gsize);
  cpu_memset(Gsize, 1, cpu_m);
  memcopy(Gsize, cpu_m, gpu_m);
  for (int i = 0; i < Gsize; ++i) {
    EXPECT_EQ(((char *) cpu_m)[i], ((char *) gpu_m)[i]);
  }
  for (int i = 0; i < Gsize; ++i) {
    EXPECT_EQ(((char *) cpu_m)[i], 1);
  }
  FreeCPU(cpu_m);
  FreeGPU(gpu_m);
}

}

