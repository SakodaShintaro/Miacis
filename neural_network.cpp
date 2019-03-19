#include"neural_network.hpp"

NeuralNetworkImpl::NeuralNetworkImpl() : device_(torch::kCUDA), conv(BLOCK_NUM, std::vector<torch::nn::Conv2d>(2, nullptr)), bn(BLOCK_NUM, std::vector<torch::nn::BatchNorm>(2, nullptr)) {
    first_conv = register_module("first_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(INPUT_CHANNEL_NUM, CHANNEL_NUM, KERNEL_SIZE).with_bias(false).padding(1)));
    first_bn = register_module("first_bn", torch::nn::BatchNorm(CHANNEL_NUM));
    for (int32_t i = 0; i < BLOCK_NUM; i++) {
        for (int32_t j = 0; j < 2; j++) {
            conv[i][j] = register_module("conv" + std::to_string(i) + "_" + std::to_string(j), torch::nn::Conv2d(torch::nn::Conv2dOptions(CHANNEL_NUM, CHANNEL_NUM, KERNEL_SIZE).with_bias(false).padding(1)));
            bn[i][j] = register_module("bn" + std::to_string(i) + "_" + std::to_string(j), torch::nn::BatchNorm(CHANNEL_NUM));
        }
    }
    policy_conv = register_module("policy_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(CHANNEL_NUM, POLICY_CHANNEL_NUM, 1).padding(0).with_bias(true)));
    value_conv = register_module("value_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(CHANNEL_NUM, CHANNEL_NUM, 1).padding(0).with_bias(false)));
    value_bn = register_module("value_bn", torch::nn::BatchNorm(CHANNEL_NUM));
    value_fc1 = register_module("value_fc1", torch::nn::Linear(SQUARE_NUM * CHANNEL_NUM, VALUE_HIDDEN_NUM));
    value_fc2 = register_module("value_fc2", torch::nn::Linear(VALUE_HIDDEN_NUM, BIN_SIZE));
}

std::pair<torch::Tensor, torch::Tensor> NeuralNetworkImpl::forward(const std::vector<float>& inputs) {
    torch::Tensor x = torch::tensor(inputs).to(device_);
    x = x.view({ -1, INPUT_CHANNEL_NUM, 9, 9 });
    x = first_conv->forward(x);
    x = first_bn->forward(x);
    x = torch::relu(x);

    for (int32_t i = 0; i < BLOCK_NUM; i++) {
        auto t = x;

        x = conv[i][0]->forward(x);
        x = bn[i][0]->forward(x);
        x = torch::relu(x);

        x = conv[i][1]->forward(x);
        x = bn[i][1]->forward(x);
        x = torch::relu(x + t);
    }

    //ここから分岐
    //policy
    torch::Tensor policy = policy_conv->forward(x);

    //value
    torch::Tensor value = value_conv->forward(x);
    value = value_bn->forward(value);
    value = torch::relu(value);
    value = value.view({ -1, SQUARE_NUM * CHANNEL_NUM });
    value = value_fc1->forward(value);
    value = torch::relu(value);
    value = value_fc2->forward(value);

#ifndef USE_CATEGORICAL
#ifdef USE_SIGMOID
    value = torch::sigmoid(value);
#else
    value = torch::tanh(value);
#endif
#endif

    return { policy, value };
}

std::pair<std::vector<PolicyType>, std::vector<ValueType>>
NeuralNetworkImpl::policyAndValueBatch(const std::vector<float>& inputs) {
    auto y = forward(inputs);

    auto batch_size = inputs.size() / (SQUARE_NUM * INPUT_CHANNEL_NUM);

    std::vector<PolicyType> policies(batch_size);
    std::vector<ValueType> values(batch_size);

    //CPUに持ってくる
    auto policy = y.first.cpu();
    auto p = policy.data<float>();
    for (int32_t i = 0; i < batch_size; i++) {
        constexpr auto unit_size = POLICY_CHANNEL_NUM * SQUARE_NUM;
        policies[i].assign(p + i * unit_size, p + (i + 1) * unit_size);
    }

#ifdef USE_CATEGORICAL
    auto value = torch::softmax(y.second, 1).cpu();
    auto value_p = value.data<float>();
    for (int32_t i = 0; i < batch_size; i++) {
        std::copy(value_p + i * BIN_SIZE, value_p + (i + 1) * BIN_SIZE, values[i].begin());
    }
#else
    //CPUに持ってくる
    auto value = y.second.cpu();
    std::copy(value.data<float>(), value.data<float>() + batch_size, values.begin());
#endif
    return { policies, values };
}

std::pair<torch::Tensor, torch::Tensor>
NeuralNetworkImpl::loss(const std::vector<float>& input,
                        const std::vector<PolicyTeacherType>& policy_teachers,
                        const std::vector<ValueTeacherType>& value_teachers) {
    auto y = forward(input);
    auto logits = y.first.view({ -1, SQUARE_NUM * POLICY_CHANNEL_NUM });

    torch::Tensor policy_target = torch::tensor(policy_teachers).to(device_);
    torch::Tensor policy_loss = torch::nll_loss(torch::log_softmax(logits, 1), policy_target);

#ifdef USE_CATEGORICAL
    torch::Tensor categorical_target = torch::tensor(value_teachers).to(device_);
    torch::Tensor value_loss = torch::nll_loss(torch::log_softmax(y.second, 1), categorical_target);
#else
    torch::Tensor value_t = torch::tensor(value_teachers).to(device_);
    torch::Tensor value = y.second.view(-1);
#ifdef USE_SIGMOID
    Var value_loss = -value_t * F::log(value) -(1 - value_t) * F::log(1 - value);
#else
    torch::Tensor value_loss = torch::mse_loss(value, value_t);
#endif
#endif
    return { policy_loss, value_loss };
}

void NeuralNetworkImpl::setGPU(int16_t gpu_id) {
    device_ = torch::Device(torch::kCUDA, gpu_id);
    to(device_);
}

NeuralNetwork nn;