#include"neural_network.hpp"

#ifdef USE_LIBTORCH

torch::Device device(torch::kCUDA);

NeuralNetworkImpl::NeuralNetworkImpl() : conv(BLOCK_NUM, std::vector<torch::nn::Conv2d>(2, nullptr)), bn(BLOCK_NUM, std::vector<torch::nn::BatchNorm>(2, nullptr)) {
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
    value = value.reshape({value.size(0), SQUARE_NUM * CHANNEL_NUM});
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

std::pair<torch::Tensor, torch::Tensor> NeuralNetworkImpl::forward(const std::vector<float>& inputs) {
    auto batch_size = inputs.size() / (INPUT_CHANNEL_NUM * SQUARE_NUM);
    torch::Tensor x = torch::tensor(inputs);
    x = x.reshape({(long)batch_size, INPUT_CHANNEL_NUM, 9, 9});
    x = x.to(device);
    return forward(x);
}

std::pair<PolicyType, ValueType> NeuralNetworkImpl::policyAndValue(const Position& pos) {
    auto y = forward(pos.makeFeature());
    auto p = y.first.cpu();
    PolicyType policy(p.data<float>(), p.data<float>() + POLICY_CHANNEL_NUM * SQUARE_NUM);

    auto value = y.second;
#ifdef USE_CATEGORICAL
    value = torch::log_softmax(value, 0).cpu();
    //std::arrayの形で返す
    ValueType retval;
    std::copy(value.data<float>(), value.data<float>() + BIN_SIZE, retval.begin());
    return { policy, retval };
#else
    return { policy, value.item<float>() };
#endif
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
    auto value = torch::log_softmax(y.second, 0).cpu();
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
NeuralNetworkImpl::loss(std::vector<float>& input, std::vector<uint32_t>& policy_labels,
                        std::vector<ValueTeacher>& value_teachers) {
    auto y = forward(input);
    auto logits = y.first.reshape({ y.first.size(0), SQUARE_NUM * POLICY_CHANNEL_NUM });

    std::vector<long> long_policy_labels(policy_labels.size());
    for (int32_t i = 0; i < policy_labels.size(); i++) {
        long_policy_labels[i] = policy_labels[i];
    }
    torch::Tensor policy_target = torch::tensor(long_policy_labels).to(device);
    torch::Tensor policy_loss = torch::nll_loss(torch::log_softmax(logits, 1), policy_target);

#ifdef USE_CATEGORICAL
    std::vector<long> long_value_labels(value_teachers.size());
    for (int32_t i = 0; i < value_teachers.size(); i++) {
        long_value_labels[i] = value_teachers[i];
    }
    auto categorical_target = torch::tensor(long_policy_labels);
    torch::Tensor value_loss = torch::nll_loss(torch::log_softmax(y.second, 1), categorical_target);
#else
    torch::Tensor value_t = torch::tensor(value_teachers).to(device);
    torch::Tensor value = y.second.reshape(y.second.size(0));
#ifdef USE_SIGMOID
    Var value_loss = -value_t * F::log(value) -(1 - value_t) * F::log(1 - value);
#else
    torch::Tensor value_loss = (value - value_t) * (value - value_t);
#endif
#endif
    return { torch::mean(policy_loss), torch::mean(value_loss) };
}

NeuralNetwork nn;

#else

template<typename Var>
BatchNormLayer<Var>::BatchNormLayer() {
    add("beta", beta_);
    add("gamma", gamma_);
    add("mean", mean_);
    add("var", var_);
}

template<typename Var>
void BatchNormLayer<Var>::init(Shape s) {
    beta_.init(s, I::Constant(0.0));
    gamma_.init(s, I::Constant(1.0));
    mean_.init(s, I::Constant(0.0));
    var_.init(s, I::Constant(1.0));
}

template<typename Var>
Node BatchNormLayer<Var>::operator()(Node x) {
    Node m = F::batch::mean(x);
    Node v = F::batch::mean((x - m) * (x - m));

    constexpr float a = 0.999;
    Tensor me = F::input_tensor(m.shape(), m.to_vector(), nullptr);
    Tensor va = F::input_tensor(v.shape(), v.to_vector(), nullptr);

    mean_.value() = a * mean_.value() + (1.0 - a) * me;
    var_.value()  = a * var_.value()  + (1.0 - a) * va;

    return F::parameter<Node>(gamma_) * (x - m) / F::sqrt(v + 1e-8) + F::parameter<Node>(beta_);
}

template<typename Var>
Tensor BatchNormLayer<Var>::operator()(Tensor x) {
    return F::parameter<Tensor>(gamma_) * (x - F::parameter<Tensor>(mean_)) /
           F::sqrt(F::parameter<Tensor>(var_) + 1e-8) + F::parameter<Tensor>(beta_);
}

//Networkの実装
template<typename Var>
NeuralNetwork<Var>::NeuralNetwork() {
    add("first_filter", first_filter);
    add("first_bn", first_bn);
    for (int32_t i = 0; i < BLOCK_NUM; i++) {
        for (int32_t j = 0; j < 2; j++) {
            add("filter" + std::to_string(i) + std::to_string(j), filter[i][j]);
            add("bn" + std::to_string(i) + std::to_string(j), bn[i][j]);
        }
    }
    add("value_filter", value_filter);
    add("value_bn", value_bn);
    add("value_pw_fc1", value_pw_fc1);
    add("value_pb_fc1", value_pb_fc1);
    add("value_pw_fc2", value_pw_fc2);
    add("value_pb_fc2", value_pb_fc2);
    add("policy_filter", policy_filter);
}

template<typename Var>
void NeuralNetwork<Var>::init() {
    first_filter.init({KERNEL_SIZE, KERNEL_SIZE, INPUT_CHANNEL_NUM, CHANNEL_NUM}, I::XavierUniformConv2D());
    first_bn.init({9, 9, CHANNEL_NUM});
    for (int32_t i = 0; i < BLOCK_NUM; i++) {
        for (int32_t j = 0; j < 2; j++) {
            filter[i][j].init({KERNEL_SIZE, KERNEL_SIZE, CHANNEL_NUM, CHANNEL_NUM}, I::XavierUniformConv2D());
            bn[i][j].init({9, 9, CHANNEL_NUM});
        }
    }
    value_filter.init({1, 1, CHANNEL_NUM, CHANNEL_NUM}, I::XavierUniformConv2D());
    value_bn.init({9, 9, CHANNEL_NUM});
    value_pw_fc1.init({VALUE_HIDDEN_NUM, SQUARE_NUM * CHANNEL_NUM}, I::XavierUniform());
    value_pb_fc1.init({VALUE_HIDDEN_NUM}, I::Constant(0));
    value_pw_fc2.init({BIN_SIZE, VALUE_HIDDEN_NUM}, I::XavierUniform());
    value_pb_fc2.init({BIN_SIZE}, I::Constant(0));
    policy_filter.init({1, 1, CHANNEL_NUM, POLICY_CHANNEL_NUM}, I::XavierUniformConv2D());
}

template<typename Var>
std::pair<Var, Var> NeuralNetwork<Var>::feedForward(const std::vector<float>& input) {
    uint32_t batch_size = (uint32_t)(input.size() / (SQUARE_NUM * INPUT_CHANNEL_NUM));
    Var x = F::input<Var>(Shape({9, 9, INPUT_CHANNEL_NUM}, batch_size), input);

    x = F::conv2d(x, F::parameter<Var>(first_filter), 1, 1, 1, 1, 1, 1);
    x = first_bn(x);
    x = F::relu(x);

    for (int32_t i = 0; i < BLOCK_NUM; i++) {
        Var t = x;

        x = F::conv2d(x, F::parameter<Var>(filter[i][0]), 1, 1, 1, 1, 1, 1);
        x = bn[i][0](x);
        x = F::relu(x);

        x = F::conv2d(x, F::parameter<Var>(filter[i][1]), 1, 1, 1, 1, 1, 1);
        x = bn[i][1](x);
        x = F::relu(x + t);
    }

    //ここから分岐
    //policy
    Var policy = F::conv2d(x, F::parameter<Var>(policy_filter), 0, 0, 1, 1, 1, 1);

    //value
    Var value = F::conv2d(x, F::parameter<Var>(value_filter), 0, 0, 1, 1, 1, 1);
    value = value_bn(value);
    value = F::relu(value);
    Var value_w_fc1 = F::parameter<Var>(value_pw_fc1);
    Var value_b_fc1 = F::parameter<Var>(value_pb_fc1);
    value = F::relu(F::matmul(value_w_fc1, F::flatten(value)) + value_b_fc1);
    Var value_w_fc2 = F::parameter<Var>(value_pw_fc2);
    Var value_b_fc2 = F::parameter<Var>(value_pb_fc2);
    value = F::matmul(value_w_fc2, value) + value_b_fc2;

#ifndef USE_CATEGORICAL
#ifdef USE_SIGMOID
    value = F::sigmoid(value);
#else
    value = F::tanh(value);
#endif
#endif
    return { policy, value };
}

template<typename Var>
std::pair<PolicyType, ValueType> NeuralNetwork<Var>::policyAndValue(const Position& pos) {
    std::vector<float> input = pos.makeFeature();
    auto y = feedForward(input);
    auto policy = F::flatten(y.first);

#ifdef USE_CATEGORICAL
    auto value = F::softmax(y.second, 0).to_vector();
        ValueType retval;
        std::copy(value.begin(), value.end(), retval.begin());
        return { policy.to_vector(), retval };
#else
    return { policy.to_vector(), y.second.to_float() };
#endif
}

template<typename Var>
std::pair<std::vector<PolicyType>, std::vector<ValueType>>
NeuralNetwork<Var>::policyAndValueBatch(const std::vector<float>& inputs) {
    auto y = feedForward(inputs);

    auto batch_size = inputs.size() / (SQUARE_NUM * INPUT_CHANNEL_NUM);

    std::vector<PolicyType> policies(batch_size);
    std::vector<ValueType> values(batch_size);

    auto policy = F::flatten(y.first).to_vector();
    for (int32_t i = 0; i < batch_size; i++) {
        policies[i].resize(POLICY_CHANNEL_NUM * SQUARE_NUM);
        for (int32_t j = 0; j < POLICY_CHANNEL_NUM * SQUARE_NUM; j++) {
            policies[i][j] = policy[i * POLICY_CHANNEL_NUM * SQUARE_NUM + j];
        }
    }

#ifdef USE_CATEGORICAL
    auto value = F::softmax(y.second, 0).to_vector();
        for (int32_t i = 0; i < batch_size; i++) {
            for (int32_t j = 0; j < BIN_SIZE; j++) {
                values[i][j] = value[i * BIN_SIZE + j];
            }
        }
#else
    values = y.second.to_vector();
#endif
    return { policies, values };
}

template<typename Var>
std::pair<Var, Var> NeuralNetwork<Var>::loss(const std::vector<float>& input,
                                             const std::vector<uint32_t>& policy_labels,
                                             const std::vector<ValueTeacher>& value_teachers) {
    auto y = feedForward(input);

    auto logits = F::flatten(y.first);

    Var policy_loss = F::softmax_cross_entropy(logits, policy_labels, 0);

#ifdef USE_CATEGORICAL
    Var value_loss = F::softmax_cross_entropy(y.second, value_teachers, 0);
#else
    const Var value_t = F::input<Var>(Shape({1}, (uint32_t)value_teachers.size()), value_teachers);
#ifdef USE_SIGMOID
    Var value_loss = -value_t * F::log(y.second) -(1 - value_t) * F::log(1 - y.second);
#else
    Var value_loss = (y.second - value_t) * (y.second - value_t);
#endif
#endif
    return { F::batch::mean(policy_loss), F::batch::mean(value_loss) };
}

template class NeuralNetwork<Node>;
template class NeuralNetwork<Tensor>;
std::unique_ptr<NeuralNetwork<Tensor>> nn(new NeuralNetwork<Tensor>);

#endif