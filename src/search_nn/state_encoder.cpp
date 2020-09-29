#include "state_encoder.hpp"

//ネットワークの設定
static constexpr int64_t BLOCK_NUM = 3;
static constexpr int32_t KERNEL_SIZE = 3;
static constexpr int32_t REDUCTION = 8;
static constexpr int32_t CHANNEL_NUM = 64;

StateEncoderImpl::StateEncoderImpl(int64_t input_channel_num) : blocks_(BLOCK_NUM, nullptr) {
    first_conv_ = register_module("first_conv_", Conv2DwithBatchNorm(input_channel_num, CHANNEL_NUM, KERNEL_SIZE));
    for (int32_t i = 0; i < BLOCK_NUM; i++) {
        blocks_[i] = register_module("blocks_" + std::to_string(i), ResidualBlock(CHANNEL_NUM, KERNEL_SIZE, REDUCTION));
    }
    last_conv_ = register_module("last_conv_", Conv2DwithBatchNorm(CHANNEL_NUM, LAST_CHANNEL_NUM, KERNEL_SIZE));
}

torch::Tensor StateEncoderImpl::forward(const torch::Tensor& x) {
    torch::Tensor y = first_conv_->forward(x);
    for (ResidualBlock& block : blocks_) {
        y = block->forward(y);
    }
    y = last_conv_->forward(y);
    return y;
}

torch::Tensor StateEncoderImpl::embed(const std::vector<float>& inputs, torch::Device device, bool fp16, bool freeze) {
    torch::Tensor x = (fp16 ? torch::tensor(inputs).to(device, torch::kHalf) : torch::tensor(inputs).to(device));
    x = x.view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
    if (freeze) {
        eval();
        torch::NoGradGuard no_grad_guard;
        x = forward(x);
    } else {
        x = forward(x);
    }
    x = torch::flatten(x, 1);
    return x;
}