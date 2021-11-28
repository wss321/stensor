/**
* Copyright 2021 wss
* Created by wss on 11月,28, 2021
*/
#include "image.hpp"
#include <gtest/gtest.h>

namespace stensor {
namespace io {
class ImageIOTest : public ::testing::Test {};

TEST_F(ImageIOTest, read2tensor) {
  std::string path = "/home/wss/CLionProjects/stensor/data/images/lena.png";
  int w = 148, h = 256;
  stensor::Tensor *t = stensor::io::read_image2tensor(path, h, w);
  EXPECT_EQ(t->shape(0), 3);
  EXPECT_EQ(t->shape(1), h);
  EXPECT_EQ(t->shape(2), w);
  std::cout << t->shape_string();

}

TEST_F(ImageIOTest, mat2tensor) {
  std::string path = "/home/wss/CLionProjects/stensor/data/images/lena.png";
  cv::Mat image = cv::imread(path, 1);
  int w = 148, h = 256;
  cv::resize(image, image, cv::Size(w, h));

  stensor::Tensor *t = stensor::io::convertMat2tensor(image, nullptr, true);
  EXPECT_EQ(t->shape(0), 3);
  EXPECT_EQ(t->shape(1), h);
  EXPECT_EQ(t->shape(2), w);
  std::cout << t->shape_string();
  cv::imshow("src", image);
  cv::waitKey(5000);
}

TEST_F(ImageIOTest, tensor2Mat) {
  std::string path = "/home/wss/CLionProjects/stensor/data/images/lena.png";
  int w = 148, h = 256;
  stensor::Tensor *t = stensor::io::read_image2tensor(path, h, w);
  cv::Mat image = stensor::io::convertTensor2Mat(t);
  EXPECT_EQ(image.channels(), 3);
  EXPECT_EQ(image.rows, h);
  EXPECT_EQ(image.cols, w);
  std::cout << t->shape_string();
  cv::imshow("src", image);
  cv::waitKey(5000);
}

}//namespace io
}//namespace stensor