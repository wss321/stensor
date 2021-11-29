/**
* Copyright 2021 wss
* Created by wss on 11月,29, 2021
*/
#ifndef STENSOR_EXC_READ_MNIST_HPP_
#define STENSOR_EXC_READ_MNIST_HPP_
#include "public/common.hpp"
#include "core/tensor.hpp"
#include "core/math_tesnsor.hpp"
#include "nn/layers.hpp"
#include "optimizer/optimizers.hpp"

using namespace std;
using namespace stensor;

int ReverseInt(int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}

nn::TensorVec read_Mnist_Label2Tensor(const string &filename, int batch_size = 64) {
  ifstream file(filename, ios::binary);
  if (file.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;
    file.read((char *) &magic_number, sizeof(magic_number));
    file.read((char *) &number_of_images, sizeof(number_of_images));
    magic_number = ReverseInt(magic_number);
    number_of_images = ReverseInt(number_of_images);
    cout << "magic number = " << magic_number << endl;
    cout << "number of images = " << number_of_images << endl;
    nn::TensorVec out(number_of_images / batch_size);
    for (int i = 0; i < out.size(); i++) {
      Tensor *out_tensor = new stensor::Tensor({batch_size, 1});
      out[i].reset(out_tensor);
      float *data = out_tensor->data();
      for (int j = 0; j < batch_size; ++j) {
        unsigned char label = 0;
        file.read((char *) &label, sizeof(label));
        *data = (float) label;
        data++;
      }

    }
    return out;
  }
  return {};
}

nn::TensorVec read_Mnist_Images_to_Tensor(const string &filename, int batch_size = 64) {
  ifstream file(filename, ios::binary);
  if (file.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    unsigned char label;
    file.read((char *) &magic_number, sizeof(magic_number));
    file.read((char *) &number_of_images, sizeof(number_of_images));
    file.read((char *) &n_rows, sizeof(n_rows));
    file.read((char *) &n_cols, sizeof(n_cols));
    magic_number = ReverseInt(magic_number);
    number_of_images = ReverseInt(number_of_images);
    n_rows = ReverseInt(n_rows);
    n_cols = ReverseInt(n_cols);

    cout << "magic number = " << magic_number << endl;
    cout << "number of images = " << number_of_images << endl;
    cout << "rows = " << n_rows << endl;
    cout << "cols = " << n_cols << endl;
    nn::TensorVec out(number_of_images / batch_size);

    for (int i = 0; i < out.size(); i++) {
      Tensor *out_tensor = new stensor::Tensor({batch_size, n_rows * n_cols});
      out[i].reset(out_tensor);
      float *data = out_tensor->data();
      for (int j = 0; j < batch_size; ++j) {
        for (int r = 0; r < n_rows; r++) {
          for (int c = 0; c < n_cols; c++) {
            unsigned char pixel = 0;
            file.read((char *) &pixel, sizeof(pixel));
            *data = (float) pixel;
            data++;
          }
        }
      }
    }
    return out;
  }
  return {};
}

#endif //STENSOR_EXC_READ_MNIST_HPP_
