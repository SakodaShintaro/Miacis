#include"neural_network.hpp"

#ifdef USE_LIBTORCH

torch::Device device(torch::kCUDA);

NeuralNetworkImpl::NeuralNetworkImpl() {
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

std::pair<torch::Tensor, torch::Tensor> NeuralNetworkImpl::forward(torch::Tensor x) {
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
        x = torch::relu(x);
    }

    //ここから分岐
    //policy
    torch::Tensor policy = policy_conv->forward(x);

    //value
    torch::Tensor value = value_conv->forward(x);
    value = value_bn->forward(value);
    value = torch::relu(value);
    value = torch::flatten(value);
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

std::pair<torch::Tensor, torch::Tensor> NeuralNetworkImpl::forward(std::vector<float>& input) {
    auto batch_size = input.size() / (INPUT_CHANNEL_NUM * SQUARE_NUM);
    torch::Tensor x = torch::tensor(input);
    x = x.reshape({(long)batch_size, INPUT_CHANNEL_NUM, 9, 9});
    x = x.to(device);
    return forward(x);
}

std::pair<PolicyType, ValueType> NeuralNetworkImpl::policyAndValue(const Position& pos) {
    std::vector<float> input = pos.makeFeature();
    auto y = forward(input);
    auto policy_data = torch::flatten(y.first).data<float>();
    PolicyType policy;
    std::copy(policy_data, policy_data + POLICY_CHANNEL_NUM * SQUARE_NUM, policy.begin());

    auto value = y.second;
#ifdef USE_CATEGORICAL
    value = torch::softmax(value, 0);
        //std::arrayの形で返す
        ValueType retval;
        std::copy(value.data<float>(), value.data<float>() + BIN_SIZE, retval.begin());
        return { policy, retval };
#else
    return { policy, value.item<float>() };
#endif
}

std::pair<std::vector<PolicyType>, std::vector<ValueType>>
NeuralNetworkImpl::policyAndValueBatch(std::vector<float>& inputs) {
    auto y = forward(inputs);

    auto batch_size = inputs.size() / (SQUARE_NUM * INPUT_CHANNEL_NUM);

    std::vector<PolicyType> policies(batch_size);
    std::vector<ValueType> values(batch_size);

    auto policy = y.first.to(torch::Device(torch::kCPU)).data<float>();
    for (int32_t i = 0; i < batch_size; i++) {
        policies[i].resize(POLICY_CHANNEL_NUM * SQUARE_NUM);
        for (int32_t j = 0; j < POLICY_CHANNEL_NUM * SQUARE_NUM; j++) {
            policies[i][j] = policy[i * POLICY_CHANNEL_NUM * SQUARE_NUM + j];
        }
    }

#ifdef USE_CATEGORICAL
    auto value = torch::softmax(y.second, 0).data<float>();
        for (int32_t i = 0; i < batch_size; i++) {
            for (int32_t j = 0; j < BIN_SIZE; j++) {
                values[i][j] = value[i * BIN_SIZE + j];
            }
        }
#else
    auto d = y.second.to(torch::Device(torch::kCPU)).data<float>();
    std::copy(d, d + batch_size, values.begin());
#endif
    return { policies, values };
}

std::pair<torch::Tensor, torch::Tensor>
NeuralNetworkImpl::loss(std::vector<float>& input, std::vector<uint32_t>& policy_labels,
                        std::vector<ValueTeacher>& value_teachers) {
    auto y = forward(input);
    auto logits = torch::flatten(y.first);

    //torch::Tensor policy_loss = torch::nll_loss();
    torch::Tensor policy_loss = logits;

#ifdef USE_CATEGORICAL
    assert(false);
        torch::Tensor value_loss;
        //Var value_loss = F::softmax_cross_entropy(y.second, value_labels, 0);
#else
    //Var value_t = F::input<Var>(Shape({1}, (uint32_t)value_teachers.size()), value_teachers);
#ifdef USE_SIGMOID
    Var value_loss = -value_t * F::log(y.second) -(1 - value_t) * F::log(1 - y.second);
#else
    //Var value_loss = (y.second - value_t) * (y.second - value_t);
    torch::Tensor value_loss = y.second;
#endif
#endif
    return { torch::mean(policy_loss), torch::mean(value_loss) };
}


#else

#endif

