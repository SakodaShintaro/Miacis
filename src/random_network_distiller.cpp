#include "random_network_distiller.hpp"
#include"include_switch.hpp"

static constexpr int32_t RANDOM_NETWORK_OUTPUT_DIM = 128;


RandomNetworkImpl::RandomNetworkImpl(int64_t input_channel_num, int64_t output_dim) {
    constexpr int64_t channel_num = 128;
    constexpr int64_t kernel_size = 3;
    conv_and_norm0_ = register_module("conv_and_norm0_", Conv2DwithBatchNorm(input_channel_num, channel_num, kernel_size));
    conv_and_norm1_ = register_module("conv_and_norm1_", Conv2DwithBatchNorm(channel_num, channel_num, kernel_size));
    conv_and_norm2_ = register_module("conv_and_norm2_", Conv2DwithBatchNorm(channel_num, channel_num, kernel_size));
    linear_ = register_module("linear_", torch::nn::Linear(torch::nn::LinearOptions(BOARD_WIDTH * BOARD_WIDTH * channel_num, output_dim).with_bias(false)));
}

torch::Tensor RandomNetworkImpl::forward(const torch::Tensor& x) {
    torch::Tensor y = conv_and_norm0_->forward(x);
    y = conv_and_norm1_->forward(y);
    y = conv_and_norm2_->forward(y);
    y = y.view({ -1, y.size(1) * y.size(2) * y.size(3) });
    y = linear_->forward(y);
    return y;
}

RandomNetworkDistillerImpl::RandomNetworkDistillerImpl() {
    random_network_target_ = register_module("random_network_target_", RandomNetwork(INPUT_CHANNEL_NUM, RANDOM_NETWORK_OUTPUT_DIM));
    random_network_infer_ = register_module("random_network_infer_", RandomNetwork(INPUT_CHANNEL_NUM, RANDOM_NETWORK_OUTPUT_DIM));
}

torch::Tensor RandomNetworkDistillerImpl::forward(const torch::Tensor& x) {
    return torch::Tensor();
}

std::vector<FloatType> RandomNetworkDistillerImpl::intrinsicValue(const std::vector<FloatType>& inputs) {
    torch::Tensor x = torch::tensor(inputs).to(device_);
    x = x.view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });

    return std::vector<FloatType>();
}