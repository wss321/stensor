/**
* Created by wss on 11月,14, 2021
*/

#include <memory>
#include <vector>

#include "common.hpp"
#include "synmem.hpp"

namespace stensor {
class SynMemTest : public ::testing::Test {};

TEST_F(SynMemTest, TestInitialization) {
  SynMem mem;
  mem.alloc_cpu(10);
  EXPECT_EQ(mem.size(), 10);
  auto p_mem = std::make_shared<SynMem>(10 * sizeof(float));
  EXPECT_EQ(p_mem->size(), 10 * sizeof(float));
}

TEST_F(SynMemTest, TestAllocationCPU) {
  SynMem mem(10);
  EXPECT_TRUE(mem.cpu_data());
}

TEST_F(SynMemTest, TestAllocationGPU) {
  SynMem mem(10);
  LOG(INFO)<<"Device:"<<mem.device();
  EXPECT_EQ(mem.device(), -1);
  mem.alloc_gpu();
  LOG(INFO)<<"Device:"<<mem.device();
}

TEST_F(SynMemTest, TestCPUWrite) {
  SynMem mem(10);
  void *cpu_data = mem.cpu_data();
  stensor::cpu_memset(mem.size(), 1, cpu_data);
  for (uint32_t i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char *>(cpu_data))[i], 1);
  }
  // do another round
  cpu_data = mem.cpu_data();
  stensor::cpu_memset(mem.size(), 2, cpu_data);
  for (uint32_t i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char *>(cpu_data))[i], 2);
  }
}

TEST_F(SynMemTest, TestGPUWrite) {
  SynMem mem(10, 0);
  void *gpu_data = mem.gpu_data();
  LOG(INFO)<<"Device:"<<mem.device();
  stensor::gpu_memset(mem.size(), 1, gpu_data);
  for (uint32_t i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char *>(gpu_data))[i], 1);
  }
  // do another round
  gpu_data = mem.gpu_data();
  stensor::gpu_memset(mem.size(), 2, gpu_data);
  for (uint32_t i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((int)(static_cast<char *>(gpu_data))[i], 2);
  }
}

TEST_F(SynMemTest, TestCPU2GPU) {
  SynMem mem(10);
  void *cpu_data = mem.cpu_data();
  stensor::cpu_memset(mem.size(), 1, cpu_data);
  for (uint32_t i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char *>(cpu_data))[i], 1);
  }
  mem.alloc_gpu();

  cpu_data = mem.cpu_data();
  stensor::cpu_memset(mem.size(), 2, cpu_data);
  for (uint32_t i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char *>(cpu_data))[i], 2);
  }
}

}