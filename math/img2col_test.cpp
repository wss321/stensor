/**
* Copyright 2021 wss
* Created by wss on 11月,28, 2021
*/
#include "math_base_cpu.hpp"
#include "public/common.hpp"
#include "core/tensor.hpp"
#include <vector>
#include <gtest/gtest.h>
namespace stensor {
class Img2ColTest : public ::testing::Test {};
TEST_F(Img2ColTest, img2colCPU) {

  int c = 1, h = 5, w = 5, sw = 1, sh = 1, pw = 0, ph = 0;
  int dh = 1, dw = 1;
  int kw = 3, kh = 3, kc = 1;
  int ow = (w - kw + 2 * pw) / sw + 1;
  int oh = (h - kh + 2 * ph) / sh + 1;

  Tensor img({c, h, w});
  for (int i = 0; i < c; ++i) {
    for (int j = 0; j < h * w; ++j) {
      img[j + i * h * w] = float(j + 1);
    }
  }

  Tensor col({c * kh * kw, oh * ow});

  std::cout << img;
  cpu_img2col(img.const_data(), c, h, w, kh, kw, ph, pw, sh, sw, dh, dw, col.data());

  std::cout << col;

  Tensor out_img({c, h, w});
  cpu_colgrad2grad(col.const_data(), c, h, w, kh, kw, ph, pw, sh, sw, dh, dw, out_img.data());
  std::cout << out_img;
}

TEST_F(Img2ColTest, img2colGPU) {

  int c = 1, h = 5, w = 5, sw = 1, sh = 1, pw = 0, ph = 0;
  int dh = 1, dw = 1;
  int kw = 3, kh = 3, kc = 1;
  int ow = (w - kw + 2 * pw) / sw + 1;
  int oh = (h - kh + 2 * ph) / sh + 1;

  Tensor img({c, h, w}, 0);
  for (int i = 0; i < c; ++i) {
    for (int j = 0; j < h * w; ++j) {
      img[j + i * h * w] = float(j + 1);
    }
  }

  Tensor col({c * kh * kw, oh * ow}, 0);

  std::cout << img;
  gpu_img2col(img.const_data(), c, h, w, kh, kw, ph, pw, sh, sw, dh, dw, col.data());
  cudaDeviceSynchronize();
  std::cout << col;

  Tensor out_img({c, h, w}, 0);
  gpu_colgrad2grad(col.const_data(), c, h, w, kh, kw, ph, pw, sh, sw, dh, dw, out_img.data());
  cudaDeviceSynchronize();
  std::cout << out_img;
}
//nd
TEST_F(Img2ColTest, ndimg2colGPU) {

  int c = 1, h = 5, w = 5, sw = 1, sh = 1, pw = 0, ph = 0;
  int dh = 1, dw = 1;
  int kw = 3, kh = 3, kc = 1;
  int ow = (w - kw + 2 * pw) / sw + 1;
  int oh = (h - kh + 2 * ph) / sh + 1;

  Tensor img({c, h, w}, 0);
  for (int i = 0; i < c; ++i) {
    for (int j = 0; j < h * w; ++j) {
      img[j + i * h * w] = float(j + 1);
    }
  }

  Tensor col({c * kh * kw, oh * ow}, 0);

  std::cout << img;
  gpu_img2col(img.const_data(), c, h, w, kh, kw, ph, pw, sh, sw, dh, dw, col.data());
  cudaDeviceSynchronize();
  std::cout << col;

  Tensor out_img({c, h, w}, 0);
  gpu_colgrad2grad(col.const_data(), c, h, w, kh, kw, ph, pw, sh, sw, dh, dw, out_img.data());
  cudaDeviceSynchronize();
  std::cout << out_img;
}
}