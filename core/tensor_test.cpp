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
  cp.copy_from(a, false, true);
}

TEST_F(TensorTest, Shape) {
  Tensor tensor(Tensor::ShapeType{3, 4});
  Tensor::Dtype *cpu_d = tensor.data();
  tensor.reshape(Tensor::ShapeType{4, 3});
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
  a.copy_from(stensor::random(a.shape()));
  std::cout << a[{-1, -1}];
  EXPECT_EQ((a[{-1, -1}]), a.data_at(-1));
}

TEST_F(TensorTest, CPUandGPU) {
  Tensor a(Tensor::ShapeType{3, 4}, 0, false);
  EXPECT_EQ(a.device(), 0);
  Tensor b(Tensor::ShapeType{3, 4});
  EXPECT_EQ(b.device(), -1);
}

TEST_F(TensorTest, SaveAndLoad) {
  Tensor a(Tensor::ShapeType{5, 5}, -1);
  a.copy_from(stensor::random(a.shape()));
  std::string path = "/home/wss/CLionProjects/stensor/output/a.pt3";
  stensor::save(a, path);
  Tensor *b = stensor::load(path);
  for (int i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i], b->data_at(i));
  }
  delete b;
}

TEST_F(TensorTest, ZeroDataGrad) {
  stensor::Config::set_random_seed(1024);
  Tensor *a = stensor::random({3, 4}, 0, 1, -1, true);
  std::cout << a;
  Tensor &b(*a);
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
  Tensor *a = stensor::random({3, 4}, 0, 1, -1, true);
  EXPECT_EQ(a->state(), CPU);
  a->to_gpu();
  EXPECT_EQ(a->state(), GPU);
  Tensor *b = stensor::random({3, 4}, 0, 1, 0, true);
  EXPECT_EQ(b->state(), GPU);
  Tensor *c = stensor::random_gaussian({3, 4}, 0, 1, 0, true);
  EXPECT_EQ(c->state(), GPU);
  delete a;
  delete b;
  delete c;
}
TEST_F(TensorTest, ElementAsinment) {
  stensor::Config::set_random_seed(1024);
  Tensor *a = stensor::random({3, 4}, 0, 1, 0, true);
  std::cout << a;
  a->zero_data();
  (*a)[0] = 1;
  std::cout << a;
  a->to_cpu();
  (*a)[2] = 1;
  std::cout << a;
  delete a;
}

int get_index(std::vector<int> &shape, Tensor::PairIndexType &stride, int index) {
  std::vector<int> new_shape(shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    new_shape[i] = stride[i].end - stride[i].start;
  }
  int num_axis = new_shape.size();
  std::vector<int> indices(shape.size());
  int div = 1;
  indices[num_axis - 1] = index % new_shape[num_axis - 1];
  for (int i = num_axis - 2; i >= 0; --i) {
    div *= new_shape[i + 1];
    indices[i] = (index / div) % new_shape[i];
  }
//  std::cout << indices << "\n";
  for (int i = 0; i < num_axis; ++i) {
    indices[i] += stride[i].start;
  }
//  std::cout << indices;
  int offset = 0;
  for (int i = 0; i < num_axis; ++i) {
    offset *= shape[i];
    if (indices.size() > i) {
      CHECK_LT(indices[i], shape[i]);
      offset += indices[i];
    }
  }
  return offset;
}

TEST_F(TensorTest, index2indices) {
  std::vector<int> shape({4, 4});
  Tensor::PairIndexType stride({{0, 3},
                                {1, 3},});
  for (int i = 0; i < 6; ++i) {
    std::cout << get_index(shape, stride, i) << " ";
  }

}

TEST_F(TensorTest, Slice) {
  stensor::Config::set_random_seed(1024);
  Tensor *a = stensor::random({3, 4}, 0, 1, 0, true);
  std::cout << a;
  std::cout<<(*a)[{{0, 2}, {2, 4}}];
  delete a;
}

}


