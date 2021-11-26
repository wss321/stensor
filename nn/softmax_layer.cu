/**
* Copyright 2021 wss
* Created by wss on 11月,24, 2021
*/
#include "softmax_layer.hpp"
#include "math/math_base_cuda.hpp"

namespace stensor {

namespace nn {

template<typename Dtype>
__global__ void softmax_backward_kernel(const int M, const int D, const int N,
                                        const Dtype *y_data,
                                        const Dtype *y_grad,
                                        Dtype beta,
                                        Dtype *x_grad) {
  CUDA_KERNEL_LOOP(index, M * D * N) {
    int n = index % N;
    int m = index / (D * N);
    Dtype sum_delta_y_mul_y = 0;
    int cur_idx = m * D * N + n;
    for (int d = 0; d < D; ++d) {
      int idx = cur_idx + d * N;
      sum_delta_y_mul_y += y_data[idx] * y_grad[idx];
    }
    if (beta == 0) {
      for (int d = 0; d < D; ++d) {
        int idx = cur_idx + d * N;
        x_grad[idx] = y_data[idx] * (Dtype(1.0) - sum_delta_y_mul_y);
      }
    } else {
      for (int d = 0; d < D; ++d) {
        int idx = cur_idx + d * N;
        x_grad[idx] = beta * x_grad[idx] + y_data[idx] * (Dtype(1.0) - sum_delta_y_mul_y);
      }
    }
  }
}

template<typename Dtype>
void gpu_softmax_backward(const int M, const int D, const int N,
                          const Dtype *y_data,
                          const Dtype *y_grad,
                          Dtype beta,
                          Dtype *x_grad) {
  softmax_backward_kernel<Dtype><<<GET_BLOCKS(M * D * N),
  CUDA_NUM_THREADS>>>(M, D, N, y_data, y_grad, beta, x_grad);
  cudaDeviceSynchronize();
  CUDA_POST_KERNEL_CHECK;
}

void Softmax::backward_gpu() {
  for (int i = 0; i < inputs_.size(); ++i) {
    SharedTensor in = inputs_[i];
    SharedTensor out = outputs_[i];
    if (!in->require_grad()||!out->require_grad()) continue;
    int caxis = in->canonical_axis_index(axis_);
    int M = in->count(0, caxis);
    int N = in->count(caxis + 1, in->num_axes());
    int D = in->shape(caxis);
    gpu_softmax_backward<float>(M, D, N, out->const_data(), out->const_grad(), 1.0f, in->grad());
  }

}
}//namespace nn

}//namespace stensor