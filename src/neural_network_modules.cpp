#include "neural_network_modules.hpp"

FCwithBatchNormImpl::FCwithBatchNormImpl(int64_t input_dim, int64_t output_dim) {
    linear_ = register_module("linear_", torch::nn::Linear(input_dim, output_dim));
    norm_ = register_module("norm_", torch::nn::BatchNorm1d(output_dim));
}

torch::Tensor FCwithBatchNormImpl::forward(const torch::Tensor& x) {
    torch::Tensor t = x;
    t = linear_->forward(t);
    t = norm_->forward(t);
    return t;
}

ResidualBlockImpl::ResidualBlockImpl(int64_t dim) {
    fc_layer0_ = register_module("fc_layer0_", FCwithBatchNorm(dim, dim));
    fc_layer1_ = register_module("fc_layer1_", FCwithBatchNorm(dim, dim));
}

torch::Tensor ResidualBlockImpl::forward(const torch::Tensor& x) {
    torch::Tensor t = x;
    t = fc_layer0_->forward(t);
    t = activation(t);
    t = fc_layer1_->forward(t);
    t = activation(x + t);
    return t;
}

torch::Tensor activation(const torch::Tensor& x) {
    //ReLU
    return torch::relu(x);

    //Mish
    //return x * torch::tanh(torch::softplus(x));

    //Swish
    //return x * torch::sigmoid(x);
}