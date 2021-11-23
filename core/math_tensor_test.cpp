/**
* Copyright 2021 wss
* Created by wss on 11月,16, 2021
*/
#include "public/common.hpp"
#include <memory>
#include "tensor.hpp"
#include <vector>
#include "math_tesnsor.hpp"
#include "transpose.hpp"

namespace stensor {
class MathTensorTest : public ::testing::Test {};
TEST_F(MathTensorTest, Gennerator) {
  Tensor tensor(Tensor::ShapeType{3, 4});
  std::cout << stensor::zeros_like(&tensor) << std::endl;
  std::cout << stensor::ones_like(&tensor) << std::endl;
  std::cout << stensor::constants_like(&tensor, 5) << std::endl;
  std::cout << stensor::ones(std::vector<int>{5, 8}) << std::endl;
  std::cout << stensor::zeros(std::vector<int>{1, 3}) << std::endl;
  std::cout << stensor::random(std::vector<int>{1, 3}) << std::endl;
  std::cout << stensor::random_gaussian(std::vector<int>{1, 3}) << std::endl;
}

TEST_F(MathTensorTest, MathUnary) {
  Tensor::ShapeType shape1{3, 4};
  Tensor *a = stensor::random(shape1);
  Tensor *b = stensor::random(shape1);
//  std::cout << "\ta:\n" << a;
//  std::cout << "\tb:\n" << b;
  auto c = stensor::add(a, b);
//  std::cout << "\ta+b:\n" << c;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(c->data_at(i), a->data_at(i) + b->data_at(i));
  }
  auto d = stensor::exp(a);
//  std::cout << "\texp:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), std::exp(a->data_at(i)));
  }

  d = stensor::pow(a, 0.0);
//  std::cout << "\tpow:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), std::pow(a->data_at(i), 0));
  }
  d = stensor::mul(a, b);
//  std::cout << "\tmul:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), a->data_at(i) * b->data_at(i));
  }
  d = stensor::div(a, b);
//  std::cout << "\tdiv:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), a->data_at(i) / b->data_at(i));
  }
  delete a;
  delete b;
  delete c;
  delete d;
}

TEST_F(MathTensorTest, MathRandomTest) {
  Tensor::ShapeType shape1{3, 4, 5};
  Tensor::ShapeType shape2{3, 1, 1};
  stensor::Config::set_random_seed(123);
  Tensor *a = stensor::random(shape1);
  stensor::Config::set_random_seed(123);
  Tensor *b = stensor::random(shape1);
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(a->data_at(i), b->data_at(i));
  }
}

TEST_F(MathTensorTest, MathBinary) {
  Tensor::ShapeType shape1{3, 4};
  Tensor *a = stensor::random(shape1);
  Tensor *b = stensor::random(shape1);
  std::cout << "\ta:\n" << a;
  std::cout << "\tb:\n" << b;

  auto d = stensor::add(a, b);
  std::cout << "\tadd:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), a->data_at(i) + b->data_at(i));
  }

  d = stensor::mul(a, b);
  std::cout << "\tmul:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), a->data_at(i) * b->data_at(i));
  }
  d = stensor::div(a, b);
  std::cout << "\tdiv:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), a->data_at(i) / b->data_at(i));
  }

  d = stensor::sub(a, b);
  std::cout << "\tsub:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), a->data_at(i) - b->data_at(i));
  }
  delete a;
  delete b;
  delete d;
}

TEST_F(MathTensorTest, MathCPUBroadCast) {
  Tensor::ShapeType shape1{3, 4, 5};
  Tensor::ShapeType shape2{3, 1, 1};
  Tensor::ShapeType shape3{3, 4, 5};
//  Tensor *a = stensor::random(shape1);
//  Tensor *b = stensor::random(shape2);
  Tensor *a = stensor::ones(shape1);
  Tensor *b = stensor::constants(shape2, 2);

  std::cout << "\ta:\n" << a;
  std::cout << "\tb:\n" << b;
  auto c = stensor::add(a, b);
  std::cout << "\ta+b:\n" << c;
  EXPECT_EQ(c->shape(), shape3);
  c = stensor::sub(a, b);
  std::cout << "\ta-b:\n" << c;
  EXPECT_EQ(c->shape(), shape3);
  c = stensor::mul(a, b);
  std::cout << "\ta*b:\n" << c;
  EXPECT_EQ(c->shape(), shape3);
  c = stensor::div(a, b);
  std::cout << "\ta/b:\n" << c;
  EXPECT_EQ(c->shape(), shape3);
}

TEST_F(MathTensorTest, MathGPUBroadCast) {
  Tensor::ShapeType shape1{3, 4, 5};
  Tensor::ShapeType shape2{4, 1};
  Tensor::ShapeType shape3{3, 4, 5};
  Tensor *a = stensor::random(shape1);
  Tensor *b = stensor::random(shape2);
//  Tensor *a = stensor::ones(shape1);
//  Tensor *b = stensor::constants(shape2, 2.5);
  a->to_gpu();
  b->to_gpu();
  EXPECT_EQ(a->device(), 0);
  std::cout << "\ta:\n" << a;
  std::cout << "\tb:\n" << b;
  Tensor *c = stensor::add(a, b);
  std::cout << "\tgpu:c=a+b:\n" << c;
  Tensor *d = stensor::sub(a, b);
  std::cout << "\tgpu:c=a-b:\n" << d;
  Tensor *e = stensor::mul(a, b);
  std::cout << "\tgpu:c=a*b:\n" << e;
  Tensor *f = stensor::div(a, b);
  std::cout << "\tgpu:c=a/b:\n" << f;
  delete a;
  delete b;
  delete c;
  delete d;
  delete e;
  delete f;

}

TEST_F(MathTensorTest, MathGPUBroadCastSpeed) {
  Tensor::ShapeType shape1{128, 3, 224, 224};
  Tensor::ShapeType shape2{3, 224, 224};
  Tensor *a = stensor::random(shape1);
  Tensor *b = stensor::random(shape2);
  long long start = systemtime_ms();
  Tensor *c1 = stensor::add(a, b);
  Tensor *d1 = stensor::sub(a, b);
  Tensor *e1 = stensor::mul(a, b);
  Tensor *f1 = stensor::div(a, b);
  LOG(INFO) << "CPU broadcast operation time:" << systemtime_ms() - start << "ms";

  a->to_gpu();
  b->to_gpu();
  EXPECT_EQ(a->device(), 0);
  start = systemtime_ms();
  Tensor *c = stensor::add(a, b);
  Tensor *d = stensor::sub(a, b);
  Tensor *e = stensor::mul(a, b);
  Tensor *f = stensor::div(a, b);
  LOG(INFO) << "GPU broadcast operation time:" << systemtime_ms() - start << "ms";

  delete a;
  delete b;
  delete c;
  delete d;
  delete e;
  delete f;
  delete c1;
  delete d1;
  delete e1;
  delete f1;

}

TEST_F(MathTensorTest, ActivateFunc) {
  Tensor::ShapeType shape1{3, 4};
  Tensor *a = stensor::random(shape1, -1.0f, 1.0f, -1);
  std::cout << "\ta:\n" << a;
  auto d = stensor::sigmoid(a);
//  std::cout << "\tsigmoid:\n" << d;
  for (int index = 0; index < a->size(); ++index) {
    float x = a->data_at(index);
    float y = d->data_at(index);
    EXPECT_EQ(1.0f / (1.0f + std::exp(-x)), y);
  }
  delete d;

  d = stensor::tanh(a);
//  std::cout << "\ttanh:\n" << d;
  for (int index = 0; index < a->size(); ++index) {
    float x = a->data_at(index);
    float y = d->data_at(index);
    EXPECT_EQ(std::tanh(x), y);
  }
  delete d;
  d = stensor::relu(a);
//  std::cout << "\trelu:\n" << d;
  for (int index = 0; index < a->size(); ++index) {
    float x = a->data_at(index);
    float y = d->data_at(index);
    EXPECT_EQ(std::max(x, 0.0f), y);
  }
  delete d;
}

TEST_F(MathTensorTest, MathMatmulTest) {
  Tensor::ShapeType shape1{3, 5000, 4000};
  Tensor::ShapeType shape2{4000, 5000};
  Tensor *a = stensor::random(shape1);
  Tensor *b = stensor::random(shape2);
  long long start = systemtime_ms();
  Tensor *c = stensor::matmul(a, b);
  LOG(INFO) << "CPU matmul operation time:" << systemtime_ms() - start << "ms";
  LOG(INFO) << "out shape:" << c->shape();

  a->to_gpu();
  b->to_gpu();
  EXPECT_EQ(a->device(), 0);
  start = systemtime_ms();
  Tensor *g = stensor::matmul(a, b);
  LOG(INFO) << "GPU matmul operation time:" << systemtime_ms() - start << "ms";

  delete a;
  delete b;
  delete g;
  delete c;

}

TEST_F(MathTensorTest, MinMax) {
  Tensor::ShapeType shape1{2, 2};
  Tensor *a = stensor::random(shape1);
  Tensor *b = stensor::random(shape1);
  Tensor *c = stensor::minimum(a, b);
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(std::min((*a)[i], (*b)[i]), (*c)[i]);
  }

  a->to_gpu();
  b->to_gpu();

  Tensor *d = stensor::minimum(a, b);
  d->to_cpu();
  bool iseq = cpu_equal(c->size(), c->data(), d->data());
  EXPECT_EQ(iseq, true);

  a->to_cpu();
  b->to_cpu();
  Tensor *e = stensor::maximum(a, b);
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(std::max((*a)[i], (*b)[i]), (*e)[i]);
  }

  a->to_gpu();
  b->to_gpu();

  Tensor *f = stensor::maximum(a, b);
  f->to_cpu();
  iseq = cpu_equal(e->size(), e->data(), f->data());
  EXPECT_EQ(iseq, true);

  delete a;
  delete b;
  delete c;
  delete d;
  delete e;
  delete f;

}

TEST_F(MathTensorTest, Repeat) {
  Tensor::ShapeType shape1{2, 1, 2};
  Tensor *a = stensor::random(shape1);
  Tensor *b = stensor::repeat(a, 1, 4);
  std::cout << a;
  std::cout << b;

  delete a;
  delete b;

  Tensor *c = stensor::random(shape1, 0);
  Tensor *d = stensor::repeat(c, 1, 4);
  std::cout << c;
  std::cout << d;

  delete c;
  delete d;
}

TEST_F(MathTensorTest, TransposeCPU) {
  Tensor::ShapeType shape1{200, 300, 400};
  Tensor *a = stensor::random(shape1, -1);
  Tensor *b = stensor::transpose(a, {1, 0, 2});
  Tensor *c = stensor::transpose(b, {1, 0, 2});

  for (int i = 0; i < c->size(); ++i) {
    EXPECT_EQ(c->data()[i], a->data()[i]);
  }
  delete a;
  delete b;
  delete c;
}
TEST_F(MathTensorTest, TransposeGPU) {
  Tensor::ShapeType shape1{200, 30, 400};
  Tensor *d = stensor::random(shape1, 0);
  Tensor *e = stensor::transpose(d, {1, 0, 2});
  Tensor *f = stensor::transpose(e, {1, 0, 2});
  for (int i = 0; i < d->size(); ++i) {
    EXPECT_EQ(f->data()[i], d->data()[i]);
  }
  delete d;
  delete e;
  delete f;
}

TEST_F(MathTensorTest, Sum) {
  Tensor::ShapeType shape1{500, 200, 400};
//  Tensor *d = stensor::random(shape1, -1);
  int axis = 1;
  Tensor *d = stensor::ones(shape1, -1);
  long long start = systemtime_ms();
  Tensor *e = stensor::sum(d, axis);
  LOG(INFO) << "CPU sum operation time:" << systemtime_ms() - start << "ms";
  std::cout << d->shape_string() << std::endl;
  std::cout << e->shape_string() << std::endl;
  for (int i = 0; i < e->size(); ++i) {
    if (i>10000) break;
    EXPECT_EQ(e->data_at(i), d->shape(axis));
  }
  d->to_gpu();
  start = systemtime_ms();
  Tensor *f = stensor::sum(d, axis);
  LOG(INFO) << "GPU sum operation time:" << systemtime_ms() - start << "ms";
  delete d;
  delete e;
  delete f;
}

TEST_F(MathTensorTest, mean) {
  Tensor::ShapeType shape1{500, 20, 400};
//  Tensor *d = stensor::random(shape1, -1);
  int axis = 1;
  Tensor *d = stensor::ones(shape1, -1);
  Tensor *e = stensor::mean(d, axis);
  std::cout << d->shape_string() << std::endl;
  std::cout << e->shape_string() << std::endl;
  for (int i = 0; i < e->size(); ++i) {
    if (i>10000) break;
    EXPECT_EQ(e->data_at(i), 1);
  }
  delete d;
  delete e;
}

TEST_F(MathTensorTest, asum) {
  Tensor::ShapeType shape1{500, 20, 400};
//  Tensor *d = stensor::random(shape1, -1);
  int axis = 1;
  Tensor *d = stensor::ones(shape1, 0);
  Tensor *e = stensor::asum(d, axis);
  std::cout << d->shape_string() << std::endl;
  std::cout << e->shape_string() << std::endl;
  for (int i = 0; i < e->size(); ++i) {
    if (i>10000) break;
    EXPECT_EQ(e->data_at(i), d->shape(axis));
  }
  delete d;
  delete e;
}

TEST_F(MathTensorTest, Concat) {
  Tensor::ShapeType shape1{50, 200, 400};
  Tensor::ShapeType shape2{10, 200, 400};

  int axis = 0;
  int device = 0;

  Tensor *a = stensor::ones(shape1, device);
  Tensor *b = stensor::constants(shape2, 2,device);
  long long start = systemtime_ms();
  Tensor *e = stensor::concat({a, b}, axis);
  std::cout << e->shape_string() << std::endl;
  LOG(INFO) << "GPU sum operation time:" << systemtime_ms() - start << "ms";
//  std::cout << e << std::endl;

  delete a;
  delete b;
  delete e;
}
}