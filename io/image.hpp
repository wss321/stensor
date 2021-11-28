/**
* Copyright 2021 wss
* Created by wss on 11月,28, 2021
*/
#ifndef STENSOR_IO_IMAGE_HPP_
#define STENSOR_IO_IMAGE_HPP_
#include "core/tensor.hpp"
#include "public/common.hpp"
#include <opencv2/opencv.hpp>
namespace stensor {
namespace io {
using namespace stensor;

Tensor *read_image2tensor(std::string &path, int h = -1, int w = -1, Tensor *out = nullptr, bool channel_first = true);
inline Tensor *read_image2tensor(std::string &path, Tensor *out = nullptr, bool channel_first = true) {
  return read_image2tensor(path, -1, -1, out, channel_first);
}

cv::Mat convertTensor2Mat(const Tensor* tensor, bool channel_first = true);
Tensor *convertMat2tensor(const cv::Mat& mat, Tensor *out = nullptr, bool channel_first = true);

}//namespace io
}//namespace stensor
#endif //STENSOR_IO_IMAGE_HPP_
