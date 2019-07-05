#include"neural_network.hpp"

//遅くなるのでオフ
//#define USE_HALF_FLOAT

NeuralNetworkImpl::NeuralNetworkImpl() : device_(torch::kCUDA),
                                         conv(BLOCK_NUM, std::vector<torch::nn::Conv2d>(2, nullptr)),
                                         bn(BLOCK_NUM, std::vector<torch::nn::BatchNorm>(2, nullptr)),
                                         fc(BLOCK_NUM, std::vector<torch::nn::Linear>(2, nullptr)) {
    first_conv = register_module("first_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(INPUT_CHANNEL_NUM, CHANNEL_NUM, KERNEL_SIZE).with_bias(false).padding(1)));
    first_bn = register_module("first_bn", torch::nn::BatchNorm(CHANNEL_NUM));
    for (int32_t i = 0; i < BLOCK_NUM; i++) {
        for (int32_t j = 0; j < 2; j++) {
            conv[i][j] = register_module("conv" + std::to_string(i) + "_" + std::to_string(j), torch::nn::Conv2d(torch::nn::Conv2dOptions(CHANNEL_NUM, CHANNEL_NUM, KERNEL_SIZE).with_bias(false).padding(1)));
            bn[i][j] = register_module("bn" + std::to_string(i) + "_" + std::to_string(j), torch::nn::BatchNorm(CHANNEL_NUM));
        }
        fc[i][0] = register_module("fc" + std::to_string(i) + "_0", torch::nn::Linear(torch::nn::LinearOptions(CHANNEL_NUM, CHANNEL_NUM / REDUCTION).with_bias(false)));
        fc[i][1] = register_module("fc" + std::to_string(i) + "_1", torch::nn::Linear(torch::nn::LinearOptions(CHANNEL_NUM / REDUCTION, CHANNEL_NUM).with_bias(false)));
    }
    policy_conv = register_module("policy_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(CHANNEL_NUM, POLICY_CHANNEL_NUM, 1).padding(0).with_bias(true)));
    value_conv = register_module("value_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(CHANNEL_NUM, CHANNEL_NUM, 1).padding(0).with_bias(false)));
    value_bn = register_module("value_bn", torch::nn::BatchNorm(CHANNEL_NUM));
    value_fc1 = register_module("value_fc1", torch::nn::Linear(SQUARE_NUM * CHANNEL_NUM, VALUE_HIDDEN_NUM));
    value_fc2 = register_module("value_fc2", torch::nn::Linear(VALUE_HIDDEN_NUM, BIN_SIZE));
}

std::pair<torch::Tensor, torch::Tensor> NeuralNetworkImpl::forward(const std::vector<float>& inputs) {
#ifdef USE_HALF_FLOAT
    torch::Tensor x = torch::tensor(inputs).to(device_, torch::kHalf);
#else
    torch::Tensor x = torch::tensor(inputs).to(device_);
#endif
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

        auto y = torch::avg_pool2d(x, {9, 9});
        y = y.view({-1, CHANNEL_NUM});
        y = fc[i][0]->forward(y);
        y = torch::relu(y);
        y = fc[i][1]->forward(y);
        y = torch::sigmoid(y);
        y = y.view({-1, CHANNEL_NUM, 1, 1});
        x = x * y;

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
    torch::Tensor policy = y.first.cpu();
#ifdef USE_HALF_FLOAT
    torch::Half* p = policy.data<torch::Half>();
#else
    float* p = policy.data<float>();
#endif
    for (int32_t i = 0; i < batch_size; i++) {
        constexpr auto unit_size = POLICY_CHANNEL_NUM * SQUARE_NUM;
        policies[i].assign(p + i * unit_size, p + (i + 1) * unit_size);
    }

#ifdef USE_CATEGORICAL
    torch::Tensor value = torch::softmax(y.second, 1).cpu();
#ifdef USE_HALF_FLOAT
    torch::Half* value_p = value.data<torch::Half>();
#else
    float* value_p = value.data<float>();
#endif
    for (int32_t i = 0; i < batch_size; i++) {
        std::copy(value_p + i * BIN_SIZE, value_p + (i + 1) * BIN_SIZE, values[i].begin());
    }
#else
    //CPUに持ってくる
    torch::Tensor value = y.second.cpu();
#ifdef USE_HALF_FLOAT
    std::copy(value.data<torch::Half>(), value.data<torch::Half>() + batch_size, values.begin());
#else
    std::copy(value.data<float>(), value.data<float>() + batch_size, values.begin());
#endif
#endif
    return { policies, values };
}

std::pair<torch::Tensor, torch::Tensor>
NeuralNetworkImpl::loss(const std::vector<float>& input,
                        const std::vector<PolicyTeacherType>& policy_teachers,
                        const std::vector<ValueTeacherType>& value_teachers) {
    auto y = forward(input);
    auto logits = y.first.view({ -1, SQUARE_NUM * POLICY_CHANNEL_NUM });

    std::vector<float> policy_dist(policy_teachers.size() * SQUARE_NUM * POLICY_CHANNEL_NUM, 0.0);
    for (int64_t i = 0; i < policy_teachers.size(); i++) {
        for (const auto& e : policy_teachers[i]) {
            policy_dist[i * SQUARE_NUM * POLICY_CHANNEL_NUM + e.first] = e.second;
        }
    }

#ifdef USE_HALF_FLOAT
    torch::Tensor policy_target = torch::tensor(policy_dist).to(device_, torch::kHalf).view({ -1, SQUARE_NUM * POLICY_CHANNEL_NUM });
#else
    torch::Tensor policy_target = torch::tensor(policy_dist).to(device_).view({ -1, SQUARE_NUM * POLICY_CHANNEL_NUM });
#endif
    //ここ怪しいので注意
    torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(logits, 1), 1, false);

#ifdef USE_CATEGORICAL
    torch::Tensor categorical_target = torch::tensor(value_teachers).to(device_);
    torch::Tensor value_loss = torch::nll_loss(torch::log_softmax(y.second, 1), categorical_target);
#else

#ifdef USE_HALF_FLOAT
    torch::Tensor value_t = torch::tensor(value_teachers).to(device_, torch::kHalf);
#else
    torch::Tensor value_t = torch::tensor(value_teachers).to(device_);
#endif
    torch::Tensor value = y.second.view(-1);
#ifdef USE_SIGMOID
    Var value_loss = -value_t * F::log(value) -(1 - value_t) * F::log(1 - value);
#else
    torch::Tensor value_loss = torch::mse_loss(value, value_t, Reduction::None);
#endif
#endif
    
    return { policy_loss, value_loss };
}

void NeuralNetworkImpl::setGPU(int16_t gpu_id) {
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
#ifdef USE_HALF_FLOAT
    for (const auto& module : modules()) {
        std::cout << module->name() << std::endl;
        if (module->name() == "torch::nn::BatchNormImpl") {
            module->to(device_);
        } else {
            module->to(device_, torch::kHalf);
        }
    }
#else
    to(device_);
#endif
}

NeuralNetwork nn;