#ifndef MIACIS_NEURAL_NETWORK_MODULES_HPP
#define MIACIS_NEURAL_NETWORK_MODULES_HPP

#include <torch/torch.h>

//畳み込みとBatchNormalizationをまとめたユニット
class FCwithBatchNormImpl : public torch::nn::Module {
public:
    FCwithBatchNormImpl(int64_t input_dim, int64_t output_dim);
    torch::Tensor forward(const torch::Tensor& x);

private:
    torch::nn::Linear linear_{ nullptr };
    torch::nn::BatchNorm1d norm_{ nullptr };
};
TORCH_MODULE(FCwithBatchNorm);

//残差ブロック:SENetの構造を利用
class ResidualBlockImpl : public torch::nn::Module {
public:
    ResidualBlockImpl(int64_t dim);
    torch::Tensor forward(const torch::Tensor& x);

private:
    FCwithBatchNorm fc_layer0_{ nullptr };
    FCwithBatchNorm fc_layer1_{ nullptr };
};
TORCH_MODULE(ResidualBlock);

torch::Tensor activation(const torch::Tensor& x);

#endif //MIACIS_NEURAL_NETWORK_MODULES_HPP