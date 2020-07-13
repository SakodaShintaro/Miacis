#ifndef MIACIS_NEURAL_NETWORK_MODULES_HPP
#define MIACIS_NEURAL_NETWORK_MODULES_HPP

#include<torch/torch.h>

//#define USE_SEPARABLE_CONV

#ifdef USE_SEPARABLE_CONV
//Separable Conv
//1回の3×3畳み込みをDepth-wise ConvとPoint-wise Convに分解することで効率化
class SeparableConvImpl : public torch::nn::Module {
public:
    SeparableConvImpl(int64_t input_ch, int64_t output_ch, int64_t kernel_size);
    torch::Tensor forward(const torch::Tensor& x);
private:
    torch::nn::Conv2d depth_wise_conv_{ nullptr };
    torch::nn::Conv2d point_wise_conv_{ nullptr };
};
TORCH_MODULE(SeparableConv);
#endif

//畳み込みとBatchNormalizationをまとめたユニット
class Conv2DwithBatchNormImpl : public torch::nn::Module {
public:
    Conv2DwithBatchNormImpl(int64_t input_ch, int64_t output_ch, int64_t kernel_size);
    torch::Tensor forward(const torch::Tensor& x);
private:
#ifdef USE_SEPARABLE_CONV
    SeparableConv        conv_{ nullptr };
#else
    torch::nn::Conv2d      conv_{ nullptr };
#endif
    torch::nn::BatchNorm2d norm_{ nullptr };
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

torch::Tensor activation(const torch::Tensor& x);

#endif //MIACIS_NEURAL_NETWORK_MODULES_HPP