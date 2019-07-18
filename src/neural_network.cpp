#include"neural_network.hpp"

//ネットワークの設定
static constexpr int32_t BLOCK_NUM = 10;
static constexpr int32_t KERNEL_SIZE = 3;
static constexpr int32_t CHANNEL_NUM = 64;
static constexpr int32_t REDUCTION = 8;
static constexpr int32_t VALUE_HIDDEN_NUM = 256;

#ifdef USE_CATEGORICAL
const std::string NeuralNetworkImpl::MODEL_PREFIX = "cat_bl" + std::to_string(BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
#else
const std::string NeuralNetworkImpl::MODEL_PREFIX = "sca_bl" + std::to_string(BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
#endif
//デフォルトで読み書きするファイル名
const std::string NeuralNetworkImpl::DEFAULT_MODEL_NAME = NeuralNetworkImpl::MODEL_PREFIX + ".model";

Conv2DwithBatchNormImpl::Conv2DwithBatchNormImpl(int64_t input_ch, int64_t output_ch, int64_t kernel_size) {
    conv_ = register_module("conv_", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_ch, output_ch, kernel_size).with_bias(false).padding(kernel_size / 2)));
    norm_ = register_module("norm_", torch::nn::BatchNorm(output_ch));
}

torch::Tensor Conv2DwithBatchNormImpl::forward(torch::Tensor& x) {
    x = conv_->forward(x);
    x = norm_->forward(x);
    return x;
}

ResidualBlockImpl::ResidualBlockImpl(int64_t channel_num, int64_t kernel_size, int64_t reduction) {
    conv_and_norm0_ = register_module("conv_and_norm0_", Conv2DwithBatchNorm(channel_num, channel_num, kernel_size));
    conv_and_norm1_ = register_module("conv_and_norm1_", Conv2DwithBatchNorm(channel_num, channel_num, kernel_size));
    linear0_ = register_module("linear0_", torch::nn::Linear(torch::nn::LinearOptions(channel_num, channel_num / reduction).with_bias(false)));
    linear1_ = register_module("linear1_", torch::nn::Linear(torch::nn::LinearOptions(channel_num / reduction, channel_num).with_bias(false)));
}

torch::Tensor ResidualBlockImpl::forward(torch::Tensor& x) {
    torch::Tensor t = x;

    x = conv_and_norm0_->forward(x);
    x = torch::relu(x);
    x = conv_and_norm1_->forward(x);

    //SENet構造
    auto y = torch::avg_pool2d(x, {9, 9});
    y = y.view({-1, CHANNEL_NUM});
    y = linear0_->forward(y);
    y = torch::relu(y);
    y = linear1_->forward(y);
    y = torch::sigmoid(y);
    y = y.view({-1, CHANNEL_NUM, 1, 1});
    x = x * y;

    x = torch::relu(x + t);
    return x;
}

NeuralNetworkImpl::NeuralNetworkImpl() : device_(torch::kCUDA), fp16_(false), state_blocks_(BLOCK_NUM, nullptr) {
    state_first_conv_and_norm_ = register_module("state_first_conv_and_norm_", Conv2DwithBatchNorm(INPUT_CHANNEL_NUM, CHANNEL_NUM, KERNEL_SIZE));
    for (int32_t i = 0; i < BLOCK_NUM; i++) {
        state_blocks_[i] = register_module("state_blocks_" + std::to_string(i), ResidualBlock(CHANNEL_NUM, KERNEL_SIZE, REDUCTION));
    }
    policy_conv_ = register_module("policy_conv_", torch::nn::Conv2d(torch::nn::Conv2dOptions(CHANNEL_NUM, POLICY_CHANNEL_NUM, 1).padding(0).with_bias(true)));
    value_conv_and_norm_ = register_module("value_conv_and_norm_", Conv2DwithBatchNorm(CHANNEL_NUM, CHANNEL_NUM, 1));
    value_linear0_ = register_module("value_linear0_", torch::nn::Linear(SQUARE_NUM * CHANNEL_NUM, VALUE_HIDDEN_NUM));
    value_linear1_ = register_module("value_linear1_", torch::nn::Linear(VALUE_HIDDEN_NUM, BIN_SIZE));
}

std::pair<torch::Tensor, torch::Tensor> NeuralNetworkImpl::forward(const std::vector<float>& inputs) {
    torch::Tensor x = (fp16_ ? torch::tensor(inputs).to(device_, torch::kHalf) : torch::tensor(inputs).to(device_));
    x = x.view({ -1, INPUT_CHANNEL_NUM, 9, 9 });
    x = state_first_conv_and_norm_->forward(x);
    x = torch::relu(x);

    for (ResidualBlock& block : state_blocks_) {
        x = block->forward(x);
    }

    //ここから分岐
    //policy
    torch::Tensor policy = policy_conv_->forward(x);

    //value
    torch::Tensor value = value_conv_and_norm_->forward(x);
    value = torch::relu(value);
    value = value.view({ -1, SQUARE_NUM * CHANNEL_NUM });
    value = value_linear0_->forward(value);
    value = torch::relu(value);
    value = value_linear1_->forward(value);

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
    if (fp16_) {
        torch::Half* p = policy.data<torch::Half>();
        for (uint64_t i = 0; i < batch_size; i++) {
            policies[i].assign(p + i * POLICY_DIM, p + (i + 1) * POLICY_DIM);
        }
    } else {
        float* p = policy.data<float>();
        for (uint64_t i = 0; i < batch_size; i++) {
            policies[i].assign(p + i * POLICY_DIM, p + (i + 1) * POLICY_DIM);
        }
    }

#ifdef USE_CATEGORICAL
    torch::Tensor value = torch::softmax(y.second, 1).cpu();
    if (fp16_) {
        torch::Half* value_p = value.data<torch::Half>();
        for (uint64_t i = 0; i < batch_size; i++) {
            std::copy(value_p + i * BIN_SIZE, value_p + (i + 1) * BIN_SIZE, values[i].begin());
        }
    } else {
        float* value_p = value.data<float>();
        for (uint64_t i = 0; i < batch_size; i++) {
            std::copy(value_p + i * BIN_SIZE, value_p + (i + 1) * BIN_SIZE, values[i].begin());
        }
    }
#else
    //CPUに持ってくる
    torch::Tensor value = y.second.cpu();
    if (fp16_) {
        std::copy(value.data<torch::Half>(), value.data<torch::Half>() + batch_size, values.begin());
    } else {
        std::copy(value.data<float>(), value.data<float>() + batch_size, values.begin());
    }
#endif
    return { policies, values };
}

std::array<torch::Tensor, LOSS_TYPE_NUM> NeuralNetworkImpl::loss(const std::vector<LearningData>& data) {
    static Position pos;
    std::vector<CalcType> inputs;
    std::vector<PolicyTeacherType> policy_teachers;
    std::vector<ValueTeacherType> value_teachers;
    for (const LearningData& datum : data) {
        pos.loadSFEN(datum.SFEN);
        const auto feature = pos.makeFeature();
        inputs.insert(inputs.end(), feature.begin(), feature.end());
        policy_teachers.push_back(datum.policy);
        value_teachers.push_back(datum.value);
    }

    auto y = forward(inputs);
    auto logits = y.first.view({ -1, POLICY_DIM });

    std::vector<float> policy_dist(policy_teachers.size() * POLICY_DIM, 0.0);
    for (uint64_t i = 0; i < policy_teachers.size(); i++) {
        for (const auto& e : policy_teachers[i]) {
            policy_dist[i * POLICY_DIM + e.first] = e.second;
        }
    }

    torch::Tensor policy_target = (fp16_ ? torch::tensor(policy_dist).to(device_, torch::kHalf) :
                                           torch::tensor(policy_dist).to(device_));

    policy_target = policy_target.view({ -1, POLICY_DIM });

    torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(logits, 1), 1, false);

#ifdef USE_CATEGORICAL
    torch::Tensor categorical_target = torch::tensor(value_teachers).to(device_);
    torch::Tensor value_loss = torch::nll_loss(torch::log_softmax(y.second, 1), categorical_target);
#else

    torch::Tensor value_t = (fp16_ ? torch::tensor(value_teachers).to(device_, torch::kHalf) :
                                     torch::tensor(value_teachers).to(device_));
    torch::Tensor value = y.second.view(-1);
#ifdef USE_SIGMOID
    torch::Tensor value_loss = -value_t * torch::log(value) - (1 - value_t) * torch::log(1 - value);
#else
    torch::Tensor value_loss = torch::mse_loss(value, value_t, Reduction::None);
#endif
#endif
    
    return { policy_loss, value_loss };
}

void NeuralNetworkImpl::setGPU(int16_t gpu_id, bool fp16) {
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    fp16_ = fp16;
    (fp16_ ? to(device_, torch::kHalf) : to(device_));
}

NeuralNetwork nn;