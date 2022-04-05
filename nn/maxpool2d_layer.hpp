//
// Created by wss on 2022/4/5.
//

#ifndef STENSOR_NN_MAXPOOL2D_LAYER_HPP_
#define STENSOR_NN_MAXPOOL2D_LAYER_HPP_

#include "module.hpp"

namespace stensor {

    namespace nn {
        class MaxPool2d : public Module {
        public:
            explicit MaxPool2d(std::string name,
                               int kernel_h, int kernel_w, int device,
                               int stride_h = 0, int stride_w = 0,
                               int padding_h = 0, int padding_w = 0) :
                    kernel_h_(kernel_h), kernel_w_(kernel_w),
                    stride_h_(stride_h), stride_w_(stride_w),
                    padding_h_(padding_h), padding_w_(padding_w) {
                name_ = std::move(name);
                type_ = "MaxPool2d";
                if (device > -1) this->state_ = GPU;
                else this->state_ = CPU;
                if (stride_h_ <= 0) stride_h_ = kernel_h_;
                if (stride_w_ <= 0) stride_w_ = kernel_w_;
            };

            ~MaxPool2d() override = default;

            TensorVec forward(TensorVec &inputs) override;

            void backward() override;

        private:
            void forward_cpu();

            void forward_gpu();

            void backward_cpu();

            void backward_gpu();

            inline std::vector<int> calc_out_shape(const std::vector<int> &in_shape) {
                // B C H W
                CHECK_EQ(in_shape.size(), 4) << "Expected shape has 4 axes:[B, C, H, W]";
                int oh = static_cast<int>(ceil(static_cast<float>(
                                                       height_ + 2 * padding_h_ - kernel_h_) / stride_h_)) + 1;
                int ow = static_cast<int>(ceil(static_cast<float>(
                                                       width_ + 2 * padding_w_ - kernel_w_) / stride_w_)) + 1;
                if (padding_h_ || padding_w_) {
                    // If we have padding, ensure that the last pooling starts strictly
                    // inside the image (instead of at the padding); otherwise clip the last.
                    if ((oh - 1) * stride_h_ >= height_ + padding_h_) {
                        --oh;
                    }
                    if ((ow - 1) * stride_w_ >= width_ + padding_w_) {
                        --ow;
                    }
                    CHECK_LT((oh - 1) * stride_h_, height_ + padding_h_);
                    CHECK_LT((ow - 1) * stride_w_, width_ + padding_w_);
                }
                return {in_shape[0], channels_, oh, ow};
            }

            int kernel_h_, kernel_w_;
            int stride_h_, stride_w_;
            int padding_w_, padding_h_;
            int channels_{};
            int height_{}, width_{};
            int pooled_height_{}, pooled_width_{};
            SharedTensor result_;
            SharedTensor max_idx_;
        DISABLE_COPY_AND_ASSIGN(MaxPool2d);
        };
    }//namespace nn

}//namespace stensor
#endif //STENSOR_NN_MAXPOOL2D_LAYER_HPP_
