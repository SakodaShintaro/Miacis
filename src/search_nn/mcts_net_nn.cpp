#include "mcts_net_nn.hpp"
#include "../include_switch.hpp"

static constexpr int64_t BLOCK_NUM = 3;
static constexpr int32_t KERNEL_SIZE = 3;
static constexpr int32_t REDUCTION = 8;
static constexpr int64_t CHANNEL_NUM = 64;
static constexpr int64_t HIDDEN_CHANNEL_NUM = 32;
static constexpr int64_t HIDDEN_DIM = BOARD_WIDTH * BOARD_WIDTH * HIDDEN_CHANNEL_NUM;

NeuralNetworksImpl::NeuralNetworksImpl() : blocks_(BLOCK_NUM, nullptr), device_(torch::kCUDA), fp16_(false) {
    simulation_policy_ = register_module("simulation_policy_", torch::nn::Linear(torch::nn::LinearOptions(HIDDEN_DIM, POLICY_DIM)));
    first_conv_ = register_module("first_conv_", Conv2DwithBatchNorm(INPUT_CHANNEL_NUM, CHANNEL_NUM, KERNEL_SIZE));
    for (int32_t i = 0; i < BLOCK_NUM; i++) {
        blocks_[i] = register_module("blocks_" + std::to_string(i), ResidualBlock(CHANNEL_NUM, KERNEL_SIZE, REDUCTION));
    }
    last_conv_ = register_module("last_conv_", Conv2DwithBatchNorm(CHANNEL_NUM, HIDDEN_CHANNEL_NUM, KERNEL_SIZE));

    backup_linear_ = register_module("backup_linear_", torch::nn::Linear(torch::nn::LinearOptions(HIDDEN_DIM * 2, HIDDEN_DIM)));

    readout_policy_ = register_module("readout_policy_", torch::nn::Linear(torch::nn::LinearOptions(HIDDEN_DIM, POLICY_DIM)));
}

torch::Tensor NeuralNetworksImpl::simulationPolicy(const torch::Tensor& h) {
    return simulation_policy_->forward(h);
}

torch::Tensor NeuralNetworksImpl::embed(const std::vector<float>& inputs) {
    torch::Tensor x = (fp16_ ? torch::tensor(inputs).to(device_, torch::kHalf) : torch::tensor(inputs).to(device_));
    x = x.view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
    return embed(x);
}

torch::Tensor NeuralNetworksImpl::embed(const torch::Tensor& x) {
    torch::Tensor y = first_conv_->forward(x);
    for (ResidualBlock& block : blocks_) {
        y = block->forward(y);
    }
    y = last_conv_->forward(y);
    y = torch::flatten(y, 1);
    return y;
}

torch::Tensor NeuralNetworksImpl::backup(const torch::Tensor& h1, const torch::Tensor& h2) {
    return h1 + backup_linear_->forward(torch::cat({ h1, h2 }, 1));
}

torch::Tensor NeuralNetworksImpl::readoutPolicy(const torch::Tensor& h) {
    return readout_policy_->forward(h);
}