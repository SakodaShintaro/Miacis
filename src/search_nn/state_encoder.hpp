#ifndef MIACIS_STATE_ENCODER_HPP
#define MIACIS_STATE_ENCODER_HPP

#include"../search_options.hpp"
#include"../include_switch.hpp"
#include<torch/torch.h>

class StateEncoderImpl : public torch::nn::Module {
public:
    StateEncoderImpl();
    torch::Tensor forward(const torch::Tensor& x);
    static constexpr int32_t LAST_CHANNEL_NUM = 32;
private:
    Conv2DwithBatchNorm first_conv_{ nullptr };
    std::vector<ResidualBlock> blocks_;
    Conv2DwithBatchNorm last_conv_{ nullptr };
};
TORCH_MODULE(StateEncoder);

#endif //MIACIS_STATE_ENCODER_HPP