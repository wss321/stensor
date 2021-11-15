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

TEST_F(SynMemTest, TestCPUWrite) {
  SynMem mem(10);
  void *cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.state(), SynMem::AT_CPU);
  std::memset(cpu_data, 1, mem.size());
  for (uint32_t i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char *>(cpu_data))[i], 1);
  }
  // do another round
  cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.state(), SynMem::AT_CPU);
  std::memset(cpu_data, 2, mem.size());
  for (uint32_t i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char *>(cpu_data))[i], 2);
  }
}

}