#include "public/common.hpp"
#include <memory>
#include "tensor.hpp"
#include <vector>
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

TEST_F(TensorTest, Gennerator) {
  Tensor tensor(Tensor::ShapeType{3, 4});
  std::cout << stensor::zeros_like(&tensor) << std::endl;
  std::cout << stensor::ones_like(&tensor) << std::endl;
  std::cout << stensor::constants_like(&tensor, 5) << std::endl;
  std::cout << stensor::ones(std::vector<uint32_t>{5, 8}) << std::endl;
  std::cout << stensor::zeros(std::vector<uint32_t>{1, 3}) << std::endl;
  std::cout << stensor::random(std::vector<uint32_t>{1, 3}) << std::endl;
  std::cout << stensor::random_gaussian(std::vector<uint32_t>{1, 3}) << std::endl;
}

TEST_F(TensorTest, Math) {
  Tensor::ShapeType shape1{3, 4};
  Tensor::ShapeType shape2{3, 4};
  Tensor a0(Tensor::ShapeType{3, 4});
  Tensor b0(Tensor::ShapeType{3, 4});
  Tensor *a = stensor::random(shape1);
  Tensor *b = stensor::random(shape1);
  std::cout << "\ta:\n" << a;
  std::cout << "\tb:\n" << b;
  auto c = stensor::add(a, b);
  std::cout << "\ta+b:\n" << c;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(c->data_at(i), a->data_at(i) + b->data_at(i));
  }
  auto d = stensor::exp(a);
  std::cout << "\texp:\n" << c;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), std::exp(a->data_at(i)));
  }

  d = stensor::pow(a, 0.0);
  std::cout << "\tpow:\n" << c;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), std::pow(a->data_at(i), 0));
  }
  d = stensor::mul(a, b);
  std::cout << "\tmul:\n" << c;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), a->data_at(i) * b->data_at(i));
  }
  d = stensor::div(a, b);
  std::cout << "\tdiv:\n" << c;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), a->data_at(i) / b->data_at(i));
  }
}

TEST_F(TensorTest, Index) {
  Tensor a(Tensor::ShapeType{3, 4});
  a.CopyFrom(stensor::random(a.shape()));
  std::cout << a[{-1, -1}];
  EXPECT_EQ((a[{-1, -1}]), a.data_at(-1));
}

TEST_F(TensorTest, SaveAndLoad) {
  Tensor a(Tensor::ShapeType{3, 4});
  a.CopyFrom(stensor::random(a.shape()));
  std::string path = "a.pt3";
  stensor::save(a, path);
  Tensor *b = stensor::load(path);
  for (int i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a.data_at(i), b->data_at(i));
  }
  std::cout << "a:" << b;
  std::cout << "b:" << b;
}

}

