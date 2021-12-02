/**
* Copyright 2021 wss
* Created by wss on 11月,21, 2021
*/

#include "transpose.hpp"
namespace stensor {

inline int indTranspose(int indY, const std::vector<int> &strideY, const std::vector<int> strideX,
                        const std::vector<int> &ordersY2X) {
  int indX = 0;
  int naxes = strideX.size();
  for (size_t ny = 0; ny < naxes; ++ny) {
    const int subY = indY / strideY[ny];
    indY -= subY * strideY[ny];
    indX += subY * strideX[ordersY2X[ny]];
  }
  return indX;
}

template<typename Dtype>
Tensor<Dtype>*transpose(Tensor<Dtype> *tensor, std::vector<int> order) {
  CHECK_GT(tensor->size(), 0) << "None of data";
  int num_axis = tensor->num_axes();
  CHECK_EQ(order.size(), num_axis);
  std::vector<int> canonical_axis(num_axis, INT16_MIN);
  std::unordered_set<int> set;
  std::vector<int> new_shape(num_axis);
  std::vector<int> stride_x(num_axis);
  std::vector<int> stride_y(num_axis);

  int temp_x = 1;
  int temp_y = 1;
  for (int i = num_axis - 1; i >= 0; --i) {
    stride_x[i] = temp_x;
    stride_y[i] = temp_y;
    canonical_axis[i] = tensor->canonical_axis_index(order[i]);
    new_shape[i] = tensor->shape(canonical_axis[i]);
    temp_x *= tensor->shape(i);
    temp_y *= new_shape[i];
    CHECK_EQ(set.count(canonical_axis[i]), 0) << "wrong order";
    set.insert(canonical_axis[i]);
  }

  Tensor<Dtype> *new_tensor = new Tensor<Dtype>(new_shape, tensor->device(), tensor->require_grad());
  if (tensor->state() == CPU) {
    Tensor<Dtype> &out = (*new_tensor);
    for (int iy = 0; iy < new_tensor->size(); ++iy) {
      int ix = indTranspose(iy, stride_y, stride_x, order);
      out[iy] = tensor->data_at(ix);
    }
  } else {
    transpose_gpu(new_shape,
                  stride_x,
                  stride_y,
                  order,
                  tensor->data(),
                  new_tensor->data());
  }

  return new_tensor;

}

}