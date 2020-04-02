#ifndef MIACIS_RANDOM_NETWORK_DISTILLER_HPP
#define MIACIS_RANDOM_NETWORK_DISTILLER_HPP

#include<torch/torch.h>
#include"neural_network.hpp"

//RND用のネットワーク
class RandomNetworkImpl : public torch::nn::Module {
public:
    RandomNetworkImpl(int64_t input_channel_num, int64_t output_dim);
    torch::Tensor forward(const torch::Tensor& x);
private:
    Conv2DwithBatchNorm conv_and_norm0_{ nullptr };
    Conv2DwithBatchNorm conv_and_norm1_{ nullptr };
    Conv2DwithBatchNorm conv_and_norm2_{ nullptr };
    torch::nn::Linear   linear_{ nullptr };
};
TORCH_MODULE(RandomNetwork);

class RandomNetworkDistillerImpl : public torch::nn::Module {
public:
    RandomNetworkDistillerImpl();
    torch::Tensor forward(const torch::Tensor& x);
    std::vector<FloatType> intrinsicValue(const std::vector<FloatType>& inputs);
private:
    RandomNetwork random_network_target_{ nullptr };
    RandomNetwork random_network_infer_{ nullptr };
};
TORCH_MODULE(RandomNetworkDistiller);

#endif //MIACIS_RANDOM_NETWORK_DISTILLER_HPP