/**
* Copyright 2021 wss
* Created by wss on 11月,28, 2021
*/
#include "image.hpp"
#include "core/transpose.hpp"

namespace stensor {
namespace io {
using namespace stensor;
Tensor *convertMat2tensor(const cv::Mat& image, Tensor *out, bool channel_first){
  CHECK(image.data) << "No image data";
  int W = image.cols;
  int H = image.rows;
  int C = image.channels();
  cv::resize(image, image, cv::Size(W, H));
  std::vector<int> shape;
  if (channel_first)
    shape = {C, H, W};
  else
    shape = {H, W, C};
  if (out == nullptr)
    out = new Tensor(shape, -1, false);
  else {
    CHECK(shape == out->shape()) << "shape mismatch:" << shape << " vs " << out->shape();
    CHECK_EQ(out->state(), CPU) << "Tensor not at CPU";
  }
  int size = out->size();
  uchar *mat_data = image.data;
  float *tensor_data = out->data();
  if (!channel_first || C != 1)
    for (int i = 0; i < size; ++i)
      tensor_data[i] = mat_data[i];
  else {
    float *out_data = out->data();
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        out_data[out->offset({0, h, w})] = *mat_data++;
        out_data[out->offset({1, h, w})] = *mat_data++;
        out_data[out->offset({2, h, w})] = *mat_data++;
      }
    }
  }
  return out;
}

Tensor *read_image2tensor(std::string &path, int new_h, int new_w, Tensor *out, bool channel_first) {
  cv::Mat image;
  image = cv::imread(path, 1);

  CHECK(image.data) << "No image data:" << path;
  int W = new_w > 0 ? new_w : image.cols;
  int H = new_h > 0 ? new_h : image.rows;
  int C = image.channels();
  cv::resize(image, image, cv::Size(W, H));
  std::vector<int> shape;
  if (channel_first)
    shape = {C, H, W};
  else
    shape = {H, W, C};
  if (out == nullptr)
    out = new Tensor(shape, -1, false);
  else {
    CHECK(shape == out->shape()) << "shape mismatch:" << shape << " vs " << out->shape();
    CHECK_EQ(out->state(), CPU) << "Tensor not at CPU";
  }
  int size = out->size();
  uchar *mat_data = image.data;
  float *tensor_data = out->data();
  if (!channel_first || C != 1)
    for (int i = 0; i < size; ++i)
      tensor_data[i] = mat_data[i];
  else {
    float *out_data = out->data();
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        out_data[out->offset({0, h, w})] = *mat_data++;
        out_data[out->offset({1, h, w})] = *mat_data++;
        out_data[out->offset({2, h, w})] = *mat_data++;
      }
    }
  }
  return out;
}

cv::Mat convertTensor2Mat(const Tensor *tensor, bool channel_first) {
  CHECK_NE(tensor, nullptr) << "None of data";
  CHECK_EQ(tensor->num_axes(), 3) << "must have 3 axes:[C, H, W] or [H, W, C]";
  if (channel_first)
    CHECK(tensor->shape(0) == 3 || tensor->shape(0) == 1)
            << "must have 3 channel or 1 channel";

  int W, H, C;
  std::vector<int> shape;
  if (channel_first) {
    C = tensor->shape(0);
    H = tensor->shape(1);
    W = tensor->shape(2);
  } else {
    H = tensor->shape(0);
    W = tensor->shape(1);
    C = tensor->shape(2);
  }

  cv::Mat image(H, W, CV_8UC(C));
  int size = tensor->size();
  uchar *mat_data = image.data;
  const float *tensor_data = tensor->const_data();
  if (!channel_first || C != 1)
    for (int i = 0; i < size; ++i)
      mat_data[i] = (uchar) tensor_data[i];
  else {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        *mat_data++ = (uchar) tensor_data[tensor->offset({0, h, w})];
        *mat_data++ = (uchar) tensor_data[tensor->offset({1, h, w})];
        *mat_data++ = (uchar) tensor_data[tensor->offset({2, h, w})];
      }
    }
  }
  return image;
}

}//namespace io
}//namespace stensor