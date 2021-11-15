#include "public/common.hpp"
#include <memory>
#include "tensor.hpp"
#include <vector>
namespace stensor {
class TensorTest : public ::testing::Test {};

TEST_F(TensorTest, Shape) {
  std::cout << "TensorTest" << std::endl;
  Tensor tensor(Tensor::ShapeType{3, 4});
  Tensor::Dtype *cpu_d = tensor.mutable_cpu_data();
  tensor.Reshape(Tensor::ShapeType{4, 3});
//  std::cout<<tensor.shape_string()<<std::endl;
  tensor.flatten();
//  std::cout<<tensor.shape_string()<<std::endl;
//  std::cout<<tensor;

  Tensor cp = tensor;
//  Tensor cp;
//  cp.CopyFrom(tensor, false, true);
  std::cout << "copy construct:" << cp;

  std::vector<uint32_t>
      out_shape = stensor::broadcast(std::vector<uint32_t>{8, 1, 6, 1},
                                     std::vector<uint32_t>{7, 1, 5});
  std::cout << out_shape << std::endl;
}

TEST_F(TensorTest, Gennerator) {
  std::cout << "TensorTest" << std::endl;
  Tensor tensor(Tensor::ShapeType{3, 4});
  std::cout << stensor::zeros_like(&tensor) << std::endl;
  std::cout << stensor::ones_like(&tensor) << std::endl;
  std::cout << stensor::ones(std::vector<uint32_t>{5, 8}) << std::endl;
  std::cout << stensor::zeros(std::vector<uint32_t>{1, 3}) << std::endl;
  std::cout << stensor::random(std::vector<uint32_t>{1, 3}) << std::endl;
  std::cout << stensor::random_gaussian(std::vector<uint32_t>{1, 3}) << std::endl;
}

}
