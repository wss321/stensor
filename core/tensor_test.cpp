#include "public/common.hpp"
#include <memory>
#include "core/tensor.hpp"
#include <vector>
#include "math_tesnsor.hpp"
#include <gtest/gtest.h>


namespace stensor {
class TensorTest : public ::testing::Test {};
TEST_F(TensorTest, Construct) {
  Tensor<float> a(Tensor<float>::ShapeType{3, 4});
  Tensor<float> cp;
  cp.copy_from(a, false, true);
}

TEST_F(TensorTest, Shape) {
  Tensor<float> tensor(Tensor<float>::ShapeType{3, 4});
  Tensor<float>::Dtype *cpu_d = tensor.data();
  tensor.reshape(Tensor<float>::ShapeType{4, 3});
  std::cout << tensor.shape_string() << std::endl;
  tensor.flatten();
  std::cout << tensor.shape_string() << std::endl;
  std::cout << tensor;

  std::vector<uint32_t> shape1{8, 1, 6, 1};
  std::vector<uint32_t> shape2{7, 1, 3};
  std::vector<uint32_t>
      out_shape = stensor::broadcast(shape1, shape2);
  std::cout << shape1 << std::endl;
  std::cout << shape2 << std::endl;
  std::cout << out_shape << std::endl;

}

TEST_F(TensorTest, Index) {
  Tensor<float> a(Tensor<float>::ShapeType{3, 4});
  a.copy_from(stensor::random<float>(a.shape()));
  std::cout << a[{-1, -1}];
  EXPECT_EQ((a[{-1, -1}]), a.data_at(-1));
}

TEST_F(TensorTest, CPUandGPU) {
  Tensor<float> a(Tensor<float>::ShapeType{3, 4}, 0, false);
  EXPECT_EQ(a.device(), 0);
  Tensor<float> b(Tensor<float>::ShapeType{3, 4});
  EXPECT_EQ(b.device(), -1);
}

TEST_F(TensorTest, SaveAndLoad) {
  Tensor<float> a(Tensor<float>::ShapeType{5, 5}, -1);
  a.copy_from(stensor::random<float>(a.shape()));
  std::string path = "/home/wss/CLionProjects/stensor/output/a.pt3";
  stensor::save(a, path);
  Tensor<float> *b = stensor::load<float>(path);
  for (int i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i], b->data_at(i));
  }
  delete b;
}

TEST_F(TensorTest, ZeroDataGrad) {
  stensor::Config::set_random_seed(1024);
  Tensor<float>* a = stensor::random<float>({3, 4}, 0, 1, -1, true);
  std::cout<<a;
  Tensor<float>& b(*a);
  a->zero_data();
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(b[i], 0);
  }
  a->zero_grad();
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(b[i], 0);
  }
  delete a;
}

TEST_F(TensorTest, FreeTest) {
  stensor::Config::set_random_seed(1024);
  Tensor<float>* a = stensor::random<float>({3, 4}, 0, 1, -1, true);
  EXPECT_EQ(a->state(), CPU);
  a->to_gpu();
  EXPECT_EQ(a->state(), GPU);
  Tensor<float>* b = stensor::random<float>({3, 4}, 0, 1, 0, true);
  EXPECT_EQ(b->state(), GPU);
  Tensor<float>* c = stensor::random_gaussian<float>({3, 4}, 0, 1, 0, true);
  EXPECT_EQ(c->state(), GPU);
  delete a;
  delete b;
  delete c;
}
TEST_F(TensorTest, ElementAsinment) {
  stensor::Config::set_random_seed(1024);
  Tensor<float>* a = stensor::random<float>({3, 4}, 0, 1, 0, true);
  std::cout<<a;
  a->zero_data();
  (*a)[0] = 1;
  std::cout<<a;
  a->to_cpu();
  (*a)[2] = 1;
  std::cout<<a;
  delete a;
}

}


