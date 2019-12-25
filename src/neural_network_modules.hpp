#ifndef MIACIS_NEURAL_NETWORK_MODULES_HPP
#define MIACIS_NEURAL_NETWORK_MODULES_HPP

#include<torch/torch.h>

//畳み込みとBatchNormalizationをまとめたユニット
class Conv2DwithBatchNormImpl : public torch::nn::Module {
public:
    Conv2DwithBatchNormImpl(int64_t input_ch, int64_t output_ch, int64_t kernel_size);
    torch::Tensor forward(const torch::Tensor& x);
private:
    torch::nn::Conv2d    conv_{ nullptr };
    torch::nn::BatchNorm norm_{ nullptr };
};
TORCH_MODULE(Conv2DwithBatchNorm);

//残差ブロック:SENetの構造を利用
class ResidualBlockImpl : public torch::nn::Module {
public:
    ResidualBlockImpl(int64_t channel_num, int64_t kernel_size, int64_t reduction);
    torch::Tensor forward(const torch::Tensor& x);
private:
    Conv2DwithBatchNorm conv_and_norm0_{ nullptr };
    Conv2DwithBatchNorm conv_and_norm1_{ nullptr };
    torch::nn::Linear   linear0_{ nullptr };
    torch::nn::Linear   linear1_{ nullptr };
};
TORCH_MODULE(ResidualBlock);


#endif //MIACIS_NEURAL_NETWORK_MODULES_HPP