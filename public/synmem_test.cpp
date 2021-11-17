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
  SynMem mem(10);
  EXPECT_EQ(mem.state(), SynMem::NONE);
  EXPECT_EQ(mem.size(), 10);
  auto p_mem = std::make_shared<SynMem>(10 * sizeof(float));
  EXPECT_EQ(p_mem->size(), 10 * sizeof(float));
}

TEST_F(SynMemTest, TestAllocationCPU) {
  SynMem mem(10);
  EXPECT_TRUE(mem.cpu_data());
  EXPECT_TRUE(mem.mutable_cpu_data());
}

TEST_F(SynMemTest, TestAllocationGPU) {
  SynMem mem(10);
  LOG(INFO)<<"Device:"<<mem.device();
  EXPECT_EQ(mem.device(), -1);
  EXPECT_TRUE(mem.gpu_data());
  EXPECT_TRUE(mem.mutable_gpu_data());
  LOG(INFO)<<"Device:"<<mem.device();
}

TEST_F(SynMemTest, TestCPUWrite) {
  SynMem mem(10);
  void *cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.state(), SynMem::AT_CPU);
  stensor::cpu_memset(mem.size(), 1, cpu_data);
  for (uint32_t i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char *>(cpu_data))[i], 1);
  }
  // do another round
  cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.state(), SynMem::AT_CPU);
  stensor::cpu_memset(mem.size(), 2, cpu_data);
  for (uint32_t i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char *>(cpu_data))[i], 2);
  }
}

TEST_F(SynMemTest, TestGPUWrite) {
  SynMem mem(10);
  void *gpu_data = mem.mutable_gpu_data();
  LOG(INFO)<<"Device:"<<mem.device();
  EXPECT_EQ(mem.state(), SynMem::AT_GPU);
  stensor::gpu_memset(mem.size(), 1, gpu_data);
  for (uint32_t i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char *>(gpu_data))[i], 1);
  }
  // do another round
  gpu_data = mem.mutable_gpu_data();
  EXPECT_EQ(mem.state(), SynMem::AT_GPU);
  stensor::gpu_memset(mem.size(), 2, gpu_data);
  for (uint32_t i = 0; i < mem.size(); ++i) {
//    LOG(INFO)<<(int)(static_cast<char *>(gpu_data))[i];
    EXPECT_EQ((int)(static_cast<char *>(gpu_data))[i], 2);
  }
}

TEST_F(SynMemTest, TestCPU2GPU) {
  SynMem mem(10);
  void *cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.state(), SynMem::AT_CPU);
  stensor::cpu_memset(mem.size(), 1, cpu_data);
  for (uint32_t i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char *>(cpu_data))[i], 1);
  }
  mem.to_gpu();
  // do another round
  cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.state(), SynMem::AT_CPU);
  stensor::cpu_memset(mem.size(), 2, cpu_data);
  for (uint32_t i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char *>(cpu_data))[i], 2);
  }
}

}