#ifndef MIACIS_STATE_ENCODER_HPP
#define MIACIS_STATE_ENCODER_HPP

#include "../include_switch.hpp"
#include "../search_options.hpp"
#include <torch/torch.h>

class StateEncoderImpl : public torch::nn::Module {
public:
    explicit StateEncoderImpl(int64_t input_channel_num = INPUT_CHANNEL_NUM);
    torch::Tensor forward(const torch::Tensor& x);
    torch::Tensor embed(const std::vector<float>& inputs, torch::Device device, bool fp16, bool freeze);
    static constexpr int32_t LAST_CHANNEL_NUM = 8;

private:
    Conv2DwithBatchNorm first_conv_{ nullptr };
    std::vector<ResidualBlock> blocks_;
    Conv2DwithBatchNorm last_conv_{ nullptr };
};
TORCH_MODULE(StateEncoder);

#endif //MIACIS_STATE_ENCODER_HPP