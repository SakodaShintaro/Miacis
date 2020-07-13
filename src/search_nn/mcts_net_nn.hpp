#ifndef MIACIS_MCTS_NET_NN_HPP
#define MIACIS_MCTS_NET_NN_HPP

#include "../neural_network_modules.hpp"

class NeuralNetworksImpl : public torch::nn::Module {
public:
    NeuralNetworksImpl();
    torch::Tensor simulationPolicy(const torch::Tensor& h);
    torch::Tensor embed(const std::vector<float>& inputs);
    torch::Tensor embed(const torch::Tensor& x);
    torch::Tensor backup(const torch::Tensor& h1, const torch::Tensor& h2);
    torch::Tensor readoutPolicy(const torch::Tensor& h);

private:
    //simulation policy network
    torch::nn::Linear simulation_policy_{ nullptr };

    //embed network
    //最初にチャンネル数を変えるConv
    Conv2DwithBatchNorm first_conv_{ nullptr };

    //同じチャンネル数で残差ブロックを通す
    std::vector<ResidualBlock> blocks_;

    //最後にまた絞るConv
    Conv2DwithBatchNorm last_conv_{ nullptr };

    //backup network
    torch::nn::Linear backup_linear_{ nullptr };

    //readout network
    torch::nn::Linear readout_policy_{ nullptr };

    //デバイスとfp16化
    torch::Device device_;
    bool fp16_;
};
TORCH_MODULE(NeuralNetworks);

#endif //MIACIS_MCTS_NET_NN_HPP