#include "public/common.hpp"
#include <memory>
#include "tensor.hpp"
#include <vector>
#include "math_tesnsor.hpp"

namespace stensor {
class TensorTest : public ::testing::Test {};
TEST_F(TensorTest, Construct) {
  Tensor a(Tensor::ShapeType{3, 4});
  Tensor cp = a;
  cp.CopyFrom(a, false, true);
}

TEST_F(TensorTest, Shape) {
  Tensor tensor(Tensor::ShapeType{3, 4});
  Tensor::Dtype *cpu_d = tensor.mutable_cpu_data();
  tensor.Reshape(Tensor::ShapeType{4, 3});
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
  Tensor a(Tensor::ShapeType{3, 4});
  a.CopyFrom(stensor::random(a.shape()));
  std::cout << a[{-1, -1}];
  EXPECT_EQ((a[{-1, -1}]), a.data_at(-1));
}
TEST_F(TensorTest, CPUandGPU) {
  Tensor a(Tensor::ShapeType{3, 4}, false, 0);
  EXPECT_EQ(a.device(), 0);
  Tensor b(Tensor::ShapeType{3, 4});
  EXPECT_EQ(b.device(), -1);
}

TEST_F(TensorTest, SaveAndLoad) {
  Tensor a(Tensor::ShapeType{300, 400});
  a.CopyFrom(stensor::zeros(a.shape()));
  std::string path = "/home/wss/CLionProjects/stensor/output/a.pt3";
  stensor::save(a, path);
  Tensor *b = stensor::load(path);
  for (int i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a.data_at(i), b->data_at(i));
  }
  std::cout << "a:\n" << b;
  std::cout << "b:\n" << b;
  delete b;
}

}

