#include "neural_network_modules.hpp"

#ifdef USE_SEPARABLE_CONV
SeparableConvImpl::SeparableConvImpl(int64_t input_ch, int64_t output_ch, int64_t kernel_size) {
    depth_wise_conv_ =
        register_module("depth_wise_conv_", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_ch, input_ch, kernel_size)
                                                                  .with_bias(false)
                                                                  .padding(kernel_size / 2)
                                                                  .groups(input_ch)));
    point_wise_conv_ = register_module(
        "point_wise_conv_", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_ch, output_ch, 1).with_bias(false).padding(0)));
}

torch::Tensor SeparableConvImpl::forward(const torch::Tensor& x) { return point_wise_conv_(depth_wise_conv_(x)); }
#endif

Conv2DwithBatchNormImpl::Conv2DwithBatchNormImpl(int64_t input_ch, int64_t output_ch, int64_t kernel_size) {
#ifdef USE_SEPARABLE_CONV
    conv_ = register_module("conv_", SeparableConv(input_ch, output_ch, kernel_size));
#else
    conv_ = register_module(
        "conv_",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(input_ch, output_ch, kernel_size).bias(false).padding(kernel_size / 2)));
#endif
    norm_ = register_module("norm_", torch::nn::BatchNorm2d(output_ch));
}

torch::Tensor Conv2DwithBatchNormImpl::forward(const torch::Tensor& x) {
    torch::Tensor t = x;
    t = conv_->forward(t);
    t = norm_->forward(t);
    return t;
}

ResidualBlockImpl::ResidualBlockImpl(int64_t channel_num, int64_t kernel_size, int64_t reduction) {
    conv_and_norm0_ = register_module("conv_and_norm0_", Conv2DwithBatchNorm(channel_num, channel_num, kernel_size));
    conv_and_norm1_ = register_module("conv_and_norm1_", Conv2DwithBatchNorm(channel_num, channel_num, kernel_size));
    linear0_ = register_module("linear0_",
                               torch::nn::Linear(torch::nn::LinearOptions(channel_num, channel_num / reduction).bias(false)));
    linear1_ = register_module("linear1_",
                               torch::nn::Linear(torch::nn::LinearOptions(channel_num / reduction, channel_num).bias(false)));
}

torch::Tensor ResidualBlockImpl::forward(const torch::Tensor& x) {
    torch::Tensor t = x;

    t = conv_and_norm0_->forward(t);
    t = activation(t);
    t = conv_and_norm1_->forward(t);

    //SENet構造
    torch::Tensor y = torch::avg_pool2d(t, { t.size(2), t.size(3) });
    y = y.view({ -1, t.size(1) });
    y = linear0_->forward(y);
    y = activation(y);
    y = linear1_->forward(y);
    y = torch::sigmoid(y);
    y = y.view({ -1, t.size(1), 1, 1 });
    t = t * y;

    t = activation(x + t);
    return t;
}

torch::Tensor activation(const torch::Tensor& x) {
    //ReLU
    //return torch::relu(x);

    //Mish
    return x * torch::tanh(torch::softplus(x));

    //Swish
    //return x * torch::sigmoid(x);
}