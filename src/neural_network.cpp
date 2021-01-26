#include "neural_network.hpp"
#include "common.hpp"
#include "include_switch.hpp"

//ネットワークの設定
#ifdef SHOGI
static constexpr int32_t BLOCK_NUM = 10;
static constexpr int32_t CHANNEL_NUM = 128;
#elif defined(OTHELLO)
static constexpr int32_t BLOCK_NUM = 5;
static constexpr int32_t CHANNEL_NUM = 64;
#endif
static constexpr int32_t KERNEL_SIZE = 3;
static constexpr int32_t REDUCTION = 8;
static constexpr int32_t VALUE_HIDDEN_NUM = 256;

#ifdef USE_CATEGORICAL
const std::string NeuralNetworkImpl::MODEL_PREFIX = "cat_bl" + std::to_string(BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
#else
const std::string NeuralNetworkImpl::MODEL_PREFIX = "sca_bl" + std::to_string(BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
#endif
//デフォルトで読み書きするファイル名
const std::string NeuralNetworkImpl::DEFAULT_MODEL_NAME = NeuralNetworkImpl::MODEL_PREFIX + ".model";

NeuralNetworkImpl::NeuralNetworkImpl() : device_(torch::kCUDA), fp16_(false), state_blocks_(BLOCK_NUM, nullptr) {
    state_first_conv_and_norm_ =
        register_module("state_first_conv_and_norm_", Conv2DwithBatchNorm(INPUT_CHANNEL_NUM, CHANNEL_NUM, KERNEL_SIZE));
    for (int32_t i = 0; i < BLOCK_NUM; i++) {
        state_blocks_[i] =
            register_module("state_blocks_" + std::to_string(i), ResidualBlock(CHANNEL_NUM, KERNEL_SIZE, REDUCTION));
    }
#ifdef REPRESENTATION_DROPOUT
    representation_dropout_ = register_module("representation_dropout_", torch::nn::Dropout2d());
#endif
    policy_conv_ = register_module(
        "policy_conv_", torch::nn::Conv2d(torch::nn::Conv2dOptions(CHANNEL_NUM, POLICY_CHANNEL_NUM, 1).padding(0).bias(true)));
    value_conv_and_norm_ = register_module("value_conv_and_norm_", Conv2DwithBatchNorm(CHANNEL_NUM, CHANNEL_NUM, 1));
    value_linear0_ = register_module("value_linear0_", torch::nn::Linear(SQUARE_NUM * CHANNEL_NUM, VALUE_HIDDEN_NUM));
    value_linear1_ = register_module("value_linear1_", torch::nn::Linear(VALUE_HIDDEN_NUM, BIN_SIZE));
}

torch::Tensor NeuralNetworkImpl::encode(const std::vector<float>& inputs) {
    torch::Tensor x = (fp16_ ? torch::tensor(inputs).to(device_, torch::kHalf) : torch::tensor(inputs).to(device_));
    x = x.view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
    x = state_first_conv_and_norm_->forward(x);
    x = activation(x);

    for (ResidualBlock& block : state_blocks_) {
        x = block->forward(x);
    }

#ifdef REPRESENTATION_DROPOUT
    x = representation_dropout_->forward(x);
#endif
    return x;
}

std::pair<torch::Tensor, torch::Tensor> NeuralNetworkImpl::decode(const torch::Tensor& representation) {
    //policy
    torch::Tensor policy = policy_conv_->forward(representation);

    //value
    torch::Tensor value = value_conv_and_norm_->forward(representation);
    value = activation(value);
    value = value.view({ -1, SQUARE_NUM * CHANNEL_NUM });
    value = value_linear0_->forward(value);
    value = activation(value);
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

std::pair<torch::Tensor, torch::Tensor> NeuralNetworkImpl::forward(const std::vector<float>& inputs) {
    return decode(encode(inputs));
}

std::pair<std::vector<PolicyType>, std::vector<ValueType>>
NeuralNetworkImpl::policyAndValueBatch(const std::vector<float>& inputs) {
    std::pair<torch::Tensor, torch::Tensor> y = forward(inputs);

    uint64_t batch_size = inputs.size() / (SQUARE_NUM * INPUT_CHANNEL_NUM);

    std::vector<PolicyType> policies(batch_size);
    std::vector<ValueType> values(batch_size);

    //CPUに持ってくる
    torch::Tensor policy = y.first.cpu();
    if (fp16_) {
        torch::Half* p = policy.data_ptr<torch::Half>();
        for (uint64_t i = 0; i < batch_size; i++) {
            policies[i].assign(p + i * POLICY_DIM, p + (i + 1) * POLICY_DIM);
        }
    } else {
        float* p = policy.data_ptr<float>();
        for (uint64_t i = 0; i < batch_size; i++) {
            policies[i].assign(p + i * POLICY_DIM, p + (i + 1) * POLICY_DIM);
        }
    }

#ifdef USE_CATEGORICAL
    torch::Tensor value = torch::softmax(y.second, 1).cpu();
    if (fp16_) {
        torch::Half* value_p = value.data_ptr<torch::Half>();
        for (uint64_t i = 0; i < batch_size; i++) {
            std::copy(value_p + i * BIN_SIZE, value_p + (i + 1) * BIN_SIZE, values[i].begin());
        }
    } else {
        float* value_p = value.data_ptr<float>();
        for (uint64_t i = 0; i < batch_size; i++) {
            std::copy(value_p + i * BIN_SIZE, value_p + (i + 1) * BIN_SIZE, values[i].begin());
        }
    }
#else
    //CPUに持ってくる
    torch::Tensor value = y.second.cpu();
    if (fp16_) {
        std::copy(value.data_ptr<torch::Half>(), value.data_ptr<torch::Half>() + batch_size, values.begin());
    } else {
        std::copy(value.data_ptr<float>(), value.data_ptr<float>() + batch_size, values.begin());
    }
#endif
    return { policies, values };
}

std::array<torch::Tensor, LOSS_TYPE_NUM> NeuralNetworkImpl::loss(const std::vector<LearningData>& data) {
    static Position pos;
    std::vector<float> inputs;
    std::vector<float> policy_teachers(data.size() * POLICY_DIM, 0.0);
    std::vector<ValueTeacherType> value_teachers;

    for (uint64_t i = 0; i < data.size(); i++) {
        pos.fromStr(data[i].position_str);

        //入力
        const std::vector<float> feature = pos.makeFeature();
        inputs.insert(inputs.end(), feature.begin(), feature.end());

        //policyの教師信号
        for (const std::pair<int32_t, float>& e : data[i].policy) {
            policy_teachers[i * POLICY_DIM + e.first] = e.second;
        }

        //valueの教師信号
        value_teachers.push_back(data[i].value);
    }

    std::pair<torch::Tensor, torch::Tensor> y = forward(inputs);
    torch::Tensor logits = y.first.view({ -1, POLICY_DIM });

    torch::Tensor policy_target =
        (fp16_ ? torch::tensor(policy_teachers).to(device_, torch::kHalf) : torch::tensor(policy_teachers).to(device_))
            .view({ -1, POLICY_DIM });

    torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(logits, 1), 1, false);

#ifdef USE_CATEGORICAL
    torch::Tensor categorical_target = torch::tensor(value_teachers).to(device_);
    torch::Tensor value_loss = torch::nll_loss(torch::log_softmax(y.second, 1), categorical_target);
#else
    torch::Tensor value_t =
        (fp16_ ? torch::tensor(value_teachers).to(device_, torch::kHalf) : torch::tensor(value_teachers).to(device_));
    torch::Tensor value = y.second.view(-1);
#ifdef USE_SIGMOID
    torch::Tensor value_loss = torch::binary_cross_entropy(value, value_t, {}, torch::Reduction::None);
#else
    torch::Tensor value_loss = torch::mse_loss(value, value_t, torch::Reduction::None);
#endif
#endif

    return { policy_loss, value_loss };
}

void NeuralNetworkImpl::setGPU(int16_t gpu_id, bool fp16) {
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    fp16_ = fp16;
    (fp16_ ? to(device_, torch::kHalf) : to(device_, torch::kFloat));
}