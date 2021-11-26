/**
* Copyright 2021 wss
* Created by wss on 11月,21, 2021
*/
#include <vector>
#include "math/math_base_cuda.hpp"
#include "public/memory_op.hpp"

namespace stensor {
template<typename Dtype>
__global__ void transpose_kernel(
    const Dtype *const X,
    const int *strideY, const int *strideX,
    const int *order, const int naxes,
    const int N,
    Dtype *Y) {
  CUDA_KERNEL_LOOP(iy, N) //for each element in Y
  {
    int indY = iy;
    int indX = 0;
    for (int ny = 0; ny < naxes; ++ny) {
      const int subY = indY / strideY[ny];
      indY -= subY * strideY[ny];
      indX += subY * strideX[order[ny]];
    }
    Y[iy] = X[indX];
  }
}

void transpose_gpu(const std::vector<int> shape,
                    const std::vector<int> &stride_x_cpu,
                    const std::vector<int> &stride_y_cpu,
                    const std::vector<int> &order,
                    const float *x,
                    float *y) {
  int num_axis = order.size();
  int stride_x[num_axis];
  int stride_y[num_axis];
  int order_arr[num_axis];

  int N = 1;
  for (int i = 0; i < num_axis; ++i) {
    stride_x[i] = stride_x_cpu[i];
    stride_y[i] = stride_y_cpu[i];
    order_arr[i] = order[i];
    N *= shape[i];
  }

  int *stride_x_gpu;
  int *stride_y_gpu;
  int *order_gpu;
  uint32_t size = num_axis * sizeof(float);
  MallocGPU((void **) &stride_x_gpu, size);
  MallocGPU((void **) &stride_y_gpu, size);
  MallocGPU((void **) &order_gpu, size);
  // copy
  stensor::gpu_copy(num_axis, stride_x, stride_x_gpu);
  stensor::gpu_copy(num_axis, stride_y, stride_y_gpu);
  stensor::gpu_copy(num_axis, order_arr, order_gpu);
  transpose_kernel<float><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      x,
      stride_x_gpu, stride_y_gpu, order_gpu, num_axis, N,
      y
  );
  CUDA_POST_KERNEL_CHECK;
  FreeGPU(stride_y_gpu);
  FreeGPU(stride_x_gpu);
  FreeGPU(order_gpu);
}
}