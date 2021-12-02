/**
* Copyright 2021 wss
* Created by wss on 11月,16, 2021
*/
#include "public/common.hpp"
#include <memory>
#include "tensor.hpp"
#include <vector>
#include "math_tesnsor.hpp"
#include "math/math_base_cpu.hpp"
#include "transpose.hpp"
#include <gtest/gtest.h>

namespace stensor {
class MathTensorTest : public ::testing::Test {};
TEST_F(MathTensorTest, Gennerator) {
  Tensor<float> tensor(std::vector<int>{3, 4});
  std::cout << stensor::zeros_like(&tensor) << std::endl;
  std::cout << stensor::ones_like(&tensor) << std::endl;
  std::cout << stensor::constants_like<float>(&tensor, 5) << std::endl;
  std::cout << stensor::ones<float>(std::vector<int>{5, 8}) << std::endl;
  std::cout << stensor::zeros<float>(std::vector<int>{1, 3}) << std::endl;
  std::cout << stensor::random<float>(std::vector<int>{1, 3}) << std::endl;
  std::cout << stensor::random_gaussian<float>(std::vector<int>{1, 3}) << std::endl;
}

TEST_F(MathTensorTest, MathUnary) {
  std::vector<int> shape1{3, 4};
  Tensor<float> *a = stensor::random<float>(shape1);
  Tensor<float> *b = stensor::random<float>(shape1);
//  std::cout << "\ta:\n" << a;
//  std::cout << "\tb:\n" << b;
  auto c = stensor::add<float>(a, b);
//  std::cout << "\ta+b:\n" << c;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(c->data_at(i), a->data_at(i) + b->data_at(i));
  }
  auto d = stensor::exp<float>(a);
//  std::cout << "\texp:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), std::exp(a->data_at(i)));
  }

  d = stensor::pow<float>(a, 0.0);
//  std::cout << "\tpow:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), std::pow(a->data_at(i), 0));
  }
  d = stensor::mul<float>(a, b);
//  std::cout << "\tmul:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), a->data_at(i) * b->data_at(i));
  }
  d = stensor::div<float>(a, b);
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
  std::vector<int> shape1{3, 4, 5};
  std::vector<int> shape2{9, 1, 1};
  stensor::Config::set_random_seed(123);
  Tensor<float> *a = stensor::random<float>(shape1);
  stensor::Config::set_random_seed(123);
  Tensor<float> *b = stensor::random<float>(shape1);
  Tensor<float> *c = stensor::random_gaussian<float>(shape2, 0);
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(a->data_at(i), b->data_at(i));
  }
}

TEST_F(MathTensorTest, MathBinary) {
  std::vector<int> shape1{3, 4};
  Tensor<float> *a = stensor::random<float>(shape1);
  Tensor<float> *b = stensor::random<float>(shape1);
  std::cout << "\ta:\n" << a;
  std::cout << "\tb:\n" << b;

  auto d = stensor::add<float>(a, b);
  std::cout << "\tadd:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), a->data_at(i) + b->data_at(i));
  }

  d = stensor::mul<float>(a, b);
  std::cout << "\tmul:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), a->data_at(i) * b->data_at(i));
  }
  d = stensor::div<float>(a, b);
  std::cout << "\tdiv:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), a->data_at(i) / b->data_at(i));
  }

  d = stensor::sub<float>(a, b);
  std::cout << "\tsub:\n" << d;
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(d->data_at(i), a->data_at(i) - b->data_at(i));
  }
  delete a;
  delete b;
  delete d;
}

TEST_F(MathTensorTest, MathCPUBroadCast) {
  std::vector<int> shape1{3, 4, 5};
  std::vector<int> shape2{3, 1, 1};
  std::vector<int> shape3{3, 4, 5};
//  Tensor<float> *a = stensor::random(shape1);
//  Tensor<float> *b = stensor::random(shape2);
  Tensor<float> *a = stensor::ones<float>(shape1);
  Tensor<float> *b = stensor::constants<float>(shape2, 2);

  std::cout << "\ta:\n" << a;
  std::cout << "\tb:\n" << b;
  auto c = stensor::add<float>(a, b);
  std::cout << "\ta+b:\n" << c;
  EXPECT_EQ(c->shape(), shape3);
  c = stensor::sub<float>(a, b);
  std::cout << "\ta-b:\n" << c;
  EXPECT_EQ(c->shape(), shape3);
  c = stensor::mul<float>(a, b);
  std::cout << "\ta*b:\n" << c;
  EXPECT_EQ(c->shape(), shape3);
  c = stensor::div<float>(a, b);
  std::cout << "\ta/b:\n" << c;
  EXPECT_EQ(c->shape(), shape3);
}

TEST_F(MathTensorTest, MathGPUBroadCast) {
  std::vector<int> shape1{3, 4, 5};
  std::vector<int> shape2{4, 1};
  std::vector<int> shape3{3, 4, 5};
  Tensor<float> *a = stensor::random<float>(shape1);
  Tensor<float> *b = stensor::random<float>(shape2);
//  Tensor<float> *a = stensor::ones(shape1);
//  Tensor<float> *b = stensor::constants(shape2, 2.5);
  a->to_gpu();
  b->to_gpu();
  EXPECT_EQ(a->device(), 0);
  std::cout << "\ta:\n" << a;
  std::cout << "\tb:\n" << b;
  Tensor<float> *c = stensor::add(a, b);
  std::cout << "\tgpu:c=a+b:\n" << c;
  Tensor<float> *d = stensor::sub(a, b);
  std::cout << "\tgpu:c=a-b:\n" << d;
  Tensor<float> *e = stensor::mul(a, b);
  std::cout << "\tgpu:c=a*b:\n" << e;
  Tensor<float> *f = stensor::div(a, b);
  std::cout << "\tgpu:c=a/b:\n" << f;
  delete a;
  delete b;
  delete c;
  delete d;
  delete e;
  delete f;

}

TEST_F(MathTensorTest, MathGPUBroadCastSpeed) {
  std::vector<int> shape1{128, 3, 224, 224};
  std::vector<int> shape2{3, 224, 224};
  Tensor<float> *a = stensor::random<float>(shape1);
  Tensor<float> *b = stensor::random<float>(shape2);
  long long start = systemtime_ms();
  Tensor<float> *c1 = stensor::add(a, b);
  Tensor<float> *d1 = stensor::sub(a, b);
  Tensor<float> *e1 = stensor::mul(a, b);
  Tensor<float> *f1 = stensor::div(a, b);
  LOG(INFO) << "CPU broadcast operation time:" << systemtime_ms() - start << "ms";

  a->to_gpu();
  b->to_gpu();
  EXPECT_EQ(a->device(), 0);
  start = systemtime_ms();
  Tensor<float> *c = stensor::add(a, b);
  Tensor<float> *d = stensor::sub(a, b);
  Tensor<float> *e = stensor::mul(a, b);
  Tensor<float> *f = stensor::div(a, b);
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
  std::vector<int> shape1{3, 4};
  Tensor<float> *a = stensor::random<float>(shape1, -1.0f, 1.0f, -1);
  std::cout << "\ta:\n" << a;
  auto d = stensor::sigmoid<float>(a);
//  std::cout << "\tsigmoid:\n" << d;
  for (int index = 0; index < a->size(); ++index) {
    float x = a->data_at(index);
    float y = d->data_at(index);
    EXPECT_EQ(1.0f / (1.0f + std::exp(-x)), y);
  }
  delete d;

  d = stensor::tanh<float>(a);
//  std::cout << "\ttanh:\n" << d;
  for (int index = 0; index < a->size(); ++index) {
    float x = a->data_at(index);
    float y = d->data_at(index);
    EXPECT_EQ(std::tanh(x), y);
  }
  delete d;
  d = stensor::relu<float>(a);
//  std::cout << "\trelu:\n" << d;
  for (int index = 0; index < a->size(); ++index) {
    float x = a->data_at(index);
    float y = d->data_at(index);
    EXPECT_EQ(std::max(x, 0.0f), y);
  }
  delete d;
}

TEST_F(MathTensorTest, MathMatmulTest) {
  std::vector<int> shape1{3, 5000, 4000};
  std::vector<int> shape2{4000, 5000};
  Tensor<float> *a = stensor::random<float>(shape1);
  Tensor<float> *b = stensor::random<float>(shape2);
  long long start = systemtime_ms();
  Tensor<float> *c = stensor::matmul<float>(a, b);
  LOG(INFO) << "CPU matmul operation time:" << systemtime_ms() - start << "ms";
  LOG(INFO) << "out shape:" << c->shape();

  a->to_gpu();
  b->to_gpu();
  EXPECT_EQ(a->device(), 0);
  start = systemtime_ms();
  Tensor<float> *g = stensor::matmul<float>(a, b);
  LOG(INFO) << "GPU matmul operation time:" << systemtime_ms() - start << "ms";

  delete a;
  delete b;
  delete g;
  delete c;

}

TEST_F(MathTensorTest, MinMax) {
  std::vector<int> shape1{2, 2};
  Tensor<float> *a = stensor::random<float>(shape1);
  Tensor<float> *b = stensor::random<float>(shape1);
  Tensor<float> *c = stensor::minimum<float>(a, b);
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(std::min((*a)[i], (*b)[i]), (*c)[i]);
  }

  a->to_gpu();
  b->to_gpu();

  Tensor<float> *d = stensor::minimum<float>(a, b);
  d->to_cpu();
  bool iseq = cpu_equal(c->size(), c->data(), d->data());
  EXPECT_EQ(iseq, true);

  a->to_cpu();
  b->to_cpu();
  Tensor<float> *e = stensor::maximum(a, b);
  for (int i = 0; i < a->size(); ++i) {
    EXPECT_EQ(std::max((*a)[i], (*b)[i]), (*e)[i]);
  }

  a->to_gpu();
  b->to_gpu();

  Tensor<float> *f = stensor::maximum(a, b);
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
  std::vector<int> shape1{2, 1, 2};
  Tensor<float> *a = stensor::random<float>(shape1);
  Tensor<float> *b = stensor::repeat(a, 1, 4);
  std::cout << a;
  std::cout << b;

  delete a;
  delete b;

  Tensor<float> *c = stensor::random<float>(shape1, 0);
  Tensor<float> *d = stensor::repeat(c, 1, 4);
  std::cout << c;
  std::cout << d;

  delete c;
  delete d;
}

TEST_F(MathTensorTest, TransposeCPU) {
  std::vector<int> shape1{200, 300, 400};
  Tensor<float> *a = stensor::random<float>(shape1, -1);
  Tensor<float> *b = stensor::transpose<float>(a, {1, 0, 2});
  Tensor<float> *c = stensor::transpose<float>(b, {1, 0, 2});

  for (int i = 0; i < c->size(); ++i) {
    EXPECT_EQ(c->data()[i], a->data()[i]);
  }
  delete a;
  delete b;
  delete c;
}
TEST_F(MathTensorTest, TransposeGPU) {
  std::vector<int> shape1{200, 30, 400};
  Tensor<float> *d = stensor::random<float>(shape1, 0);
  Tensor<float> *e = stensor::transpose(d, {1, 0, 2});
  Tensor<float> *f = stensor::transpose(e, {1, 0, 2});
  for (int i = 0; i < d->size(); ++i) {
    EXPECT_EQ(f->data()[i], d->data()[i]);
  }
  delete d;
  delete e;
  delete f;
}

TEST_F(MathTensorTest, Sum) {
  std::vector<int> shape1{500, 200, 400};
//  Tensor<float> *d = stensor::random(shape1, -1);
  int axis = 1;
  Tensor<float> *d = stensor::ones<float>(shape1, -1);
  long long start = systemtime_ms();
  Tensor<float> *e = stensor::sum(d, axis);
  LOG(INFO) << "CPU sum operation time:" << systemtime_ms() - start << "ms";
  std::cout << d->shape_string() << std::endl;
  std::cout << e->shape_string() << std::endl;
  for (int i = 0; i < e->size(); ++i) {
    if (i>10000) break;
    EXPECT_EQ(e->data_at(i), d->shape(axis));
  }
  d->to_gpu();
  start = systemtime_ms();
  Tensor<float> *f = stensor::sum(d, axis);
  LOG(INFO) << "GPU sum operation time:" << systemtime_ms() - start << "ms";
  delete d;
  delete e;
  delete f;
}

TEST_F(MathTensorTest, mean) {
  std::vector<int> shape1{500, 20, 400};
//  Tensor<float> *d = stensor::random(shape1, -1);
  int axis = 1;
  Tensor<float> *d = stensor::ones<float>(shape1, -1);
  Tensor<float> *e = stensor::mean(d, axis);
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
  std::vector<int> shape1{500, 20, 400};
//  Tensor<float> *d = stensor::random(shape1, -1);
  int axis = 1;
  Tensor<float> *d = stensor::ones<float>(shape1, 0);
  Tensor<float> *e = stensor::asum(d, axis);
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
  std::vector<int> shape1{50, 2000, 400};
  std::vector<int> shape2{10, 2000, 400};

  int axis = 0;
  int device = 0;

  Tensor<float> *a = stensor::ones<float>(shape1, device);
  Tensor<float> *b = stensor::constants<float>(shape2, 2,device);
  long long start = systemtime_ms();
  Tensor<float> *e = stensor::concat<float>({a, b}, axis);
  std::cout << e->shape_string() << std::endl;
  LOG(INFO) << "GPU concat operation time:" << systemtime_ms() - start << "ms";

  a->to_cpu();
  b->to_cpu();
  start = systemtime_ms();
  Tensor<float> *f = stensor::concat<float>({a, b}, axis);
  std::cout << f->shape_string() << std::endl;
  LOG(INFO) << "CPU concat operation time:" << systemtime_ms() - start << "ms";

  delete a;
  delete b;
  delete e;
  delete f;
}

TEST_F(MathTensorTest, Softmax) {
  std::vector<int> shape1{5, 4};

  int axis = -1;
  int device = 0;

  Tensor<float> *a = stensor::ones<float>(shape1, device);
  long long start = systemtime_ms();
  Tensor<float> *e = stensor::softmax(a, axis);
  std::cout << e->shape_string() << std::endl;
  LOG(INFO) << "GPU softmax operation time:" << systemtime_ms() - start << "ms";
  cudaDeviceSynchronize();
  std::cout<<e;

  a->to_cpu();
  start = systemtime_ms();
  e = stensor::softmax(a, axis);
  std::cout << e->shape_string() << std::endl;
//  std::cout<<e;
  LOG(INFO) << "CPU softmax operation time:" << systemtime_ms() - start << "ms";
  //  std::cout << e << std::endl;

  delete a;
  delete e;
}

TEST_F(MathTensorTest, OneHot) {
  std::vector<int> shape1{10};

  int device = 0;

  Tensor<float> *a = stensor::zeros<float>(shape1, device);
  for (int i = 0; i < a->size(); ++i) {
    (*a)[i] = float(i);
  }
  std::cout << a << std::endl;
  Tensor<float> *e = stensor::one_hot(a, 10);
  std::cout << e << std::endl;

  a->to_cpu();
  e = stensor::one_hot(a, 10);
  std::cout << e << std::endl;


  delete a;
  delete e;
}
TEST_F(MathTensorTest, Argmax) {
  std::vector<int> shape1{5, 4};

  int axis = -1;
  int device = 0;

  Tensor<float> *a = stensor::random<float>(shape1, device);
  cudaDeviceSynchronize();
  std::cout<<a;
  long long start = systemtime_ms();
  Tensor<float> *e = stensor::argmax(a, axis);
  std::cout << e->shape_string() << std::endl;
  LOG(INFO) << "GPU softmax operation time:" << systemtime_ms() - start << "ms";
  cudaDeviceSynchronize();
  std::cout<<e;

  a->to_cpu();
  start = systemtime_ms();
  e = stensor::argmax(a, axis);
  std::cout << e->shape_string() << std::endl;
  std::cout<<e;
  LOG(INFO) << "CPU softmax operation time:" << systemtime_ms() - start << "ms";
  //  std::cout << e << std::endl;

  delete a;
  delete e;
}
}