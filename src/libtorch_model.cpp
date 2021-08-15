#include "libtorch_model.hpp"
#include "common.hpp"
#include "include_switch.hpp"
#include "learn.hpp"

using namespace torch::nn;

//ネットワークの設定
#ifdef SHOGI
static constexpr int32_t BLOCK_NUM = 10;
static constexpr int32_t CHANNEL_NUM = 256;
#elif defined(OTHELLO)
static constexpr int32_t BLOCK_NUM = 5;
static constexpr int32_t CHANNEL_NUM = 64;
#elif defined(GO)
static constexpr int32_t BLOCK_NUM = 5;
static constexpr int32_t CHANNEL_NUM = 64;
#endif
static constexpr int64_t KERNEL_SIZE = 3;
static constexpr int64_t REDUCTION = 8;
static constexpr int64_t VALUE_HIDDEN_NUM = 256;

torch::Tensor activation(const torch::Tensor& x) {
    //ReLU
    return torch::relu(x);

    //Mish
    //return x * torch::tanh(torch::softplus(x));

    //Swish
    //return x * torch::sigmoid(x);
}

Conv2DwithBatchNormImpl::Conv2DwithBatchNormImpl(int64_t input_ch, int64_t output_ch, int64_t kernel_size) {
    conv_ =
        register_module("conv_", Conv2d(Conv2dOptions(input_ch, output_ch, kernel_size).bias(false).padding(kernel_size / 2)));
    norm_ = register_module("norm_", BatchNorm2d(output_ch));
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
    linear0_ = register_module("linear0_", Linear(LinearOptions(channel_num, channel_num / reduction).bias(false)));
    linear1_ = register_module("linear1_", Linear(LinearOptions(channel_num / reduction, channel_num).bias(false)));
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

ValueHeadImpl::ValueHeadImpl() {
    conv_and_norm_ = register_module("conv_and_norm_", Conv2DwithBatchNorm(CHANNEL_NUM, CHANNEL_NUM, 1));
    linear0_ = register_module("linear0_", Linear(CHANNEL_NUM, VALUE_HIDDEN_NUM));
    linear1_ = register_module("linear1_", Linear(VALUE_HIDDEN_NUM, BIN_SIZE));
}

torch::Tensor ValueHeadImpl::forward(const torch::Tensor& x) {
    torch::Tensor value = conv_and_norm_->forward(x);
    value = activation(value);
    value = torch::avg_pool2d(value, { value.size(2), value.size(3) });
    value = value.view({ -1, CHANNEL_NUM });
    value = linear0_->forward(value);
    value = activation(value);
    value = linear1_->forward(value);
    return value;
}

NetworkImpl::NetworkImpl() : blocks_(BLOCK_NUM, nullptr) {
    first_conv_and_norm_ =
        register_module("first_conv_and_norm_", Conv2DwithBatchNorm(INPUT_CHANNEL_NUM, CHANNEL_NUM, KERNEL_SIZE));
    for (int32_t i = 0; i < BLOCK_NUM; i++) {
        blocks_[i] = register_module("blocks_" + std::to_string(i), ResidualBlock(CHANNEL_NUM, KERNEL_SIZE, REDUCTION));
    }
    policy_head_ =
        register_module("policy_head_", Conv2d(Conv2dOptions(CHANNEL_NUM, POLICY_CHANNEL_NUM, 1).padding(0).bias(true)));
    value_head_ = register_module("value_head_", ValueHead());
}

torch::Tensor NetworkImpl::encode(const torch::Tensor& x) {
    torch::Tensor r = x;
    r = first_conv_and_norm_->forward(x);
    r = activation(x);

    for (ResidualBlock& block : blocks_) {
        r = block->forward(x);
    }

    return r;
}

std::pair<torch::Tensor, torch::Tensor> NetworkImpl::decode(const torch::Tensor& representation) {
    //policy
    torch::Tensor policy = policy_head_->forward(representation);

    //value
    torch::Tensor value = value_head_->forward(representation);

#ifndef USE_CATEGORICAL
#ifdef USE_SIGMOID
    value = torch::sigmoid(value);
#else
    value = torch::tanh(value);
#endif
#endif

    return { policy, value };
}

std::pair<torch::Tensor, torch::Tensor> NetworkImpl::forward(const torch::Tensor& x) { return decode(encode(x)); }

void LibTorchModel::load(const std::string& model_path, int64_t gpu_id) {
    torch::load(network_, model_path);
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    network_->to(device_);
}

void LibTorchModel::save(const std::string& model_path) { torch::save(network_, model_path); }

std::array<torch::Tensor, LOSS_TYPE_NUM> LibTorchModel::loss(const std::vector<LearningData>& data) {
    auto [input, policy_target, value_target] = learningDataToTensor(data, device_, false);
    auto out = network_->forward(input);
    torch::Tensor policy = out.first;
    torch::Tensor value = out.second;

    torch::Tensor policy_logits = policy.view({ -1, POLICY_DIM });
    torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(policy_logits, 1), 1, false);

#ifdef USE_CATEGORICAL
    torch::Tensor value_loss = torch::nll_loss(torch::log_softmax(value, 1), value_target);
#else
    value = value.view(-1);
#ifdef USE_SIGMOID
    torch::Tensor value_loss = torch::binary_cross_entropy(value, value_target, {}, torch::Reduction::None);
#else
    torch::Tensor value_loss = torch::mse_loss(value, value_target, torch::Reduction::None);
#endif
#endif

    return { policy_loss, value_loss };
}

std::array<torch::Tensor, LOSS_TYPE_NUM> LibTorchModel::validLoss(const std::vector<LearningData>& data) {
#ifdef USE_CATEGORICAL
    auto [input, policy_target, value_target] = learningDataToTensor(data, device_, true);
    auto out = network_->forward(input);
    torch::Tensor policy_logit = out.first;
    torch::Tensor value_logit = out.second;

    torch::Tensor logits = policy_logit.view({ -1, POLICY_DIM });

    torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(logits, 1), 1, false);

    //Valueの分布を取得
    torch::Tensor value_cat = torch::softmax(value_logit, 1);

    //i番目の要素が示す値はMIN_SCORE + (i + 0.5) * VALUE_WIDTH
    std::vector<float> each_value;
    for (int64_t i = 0; i < BIN_SIZE; i++) {
        each_value.emplace_back(MIN_SCORE + (i + 0.5) * VALUE_WIDTH);
    }
    torch::Tensor each_value_tensor = torch::tensor(each_value).to(device_);

    //Categorical分布と内積を取ることで期待値を求める
    torch::Tensor value = (each_value_tensor * value_cat).sum(1);

#ifdef USE_SIGMOID
    torch::Tensor value_loss = torch::binary_cross_entropy(value, value_target, {}, torch::Reduction::None);
#else
    torch::Tensor value_loss = torch::mse_loss(value, value_target, torch::Reduction::None);
#endif

    return { policy_loss, value_loss };
#else
    //Scalarモデルの場合はloss関数と同じ
    return loss(data);
#endif
}

std::array<torch::Tensor, LOSS_TYPE_NUM> LibTorchModel::mixUpLoss(const std::vector<LearningData>& data, float alpha) {
    auto [input_tensor, policy_target, value_target] = learningDataToTensor(data, device_, false);

    //混合比率の振り出し
    std::gamma_distribution<float> gamma_dist(alpha);
    float gamma1 = gamma_dist(engine), gamma2 = gamma_dist(engine);
    float beta = gamma1 / (gamma1 + gamma2);

    //データのmixup
    input_tensor = beta * input_tensor + (1 - beta) * input_tensor.roll(1, 0);
    policy_target = beta * policy_target + (1 - beta) * policy_target.roll(1, 0);
    value_target = beta * value_target + (1 - beta) * value_target.roll(1, 0);

    auto out = network_->forward(input_tensor);
    torch::Tensor policy = out.first;
    torch::Tensor value = out.second;

    torch::Tensor policy_logits = policy.view({ -1, POLICY_DIM });

    torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(policy_logits, 1), 1, false);

#ifdef USE_CATEGORICAL
    torch::Tensor value_loss = torch::nll_loss(torch::log_softmax(value, 1), value_target);
#else
    value = value.view(-1);
#ifdef USE_SIGMOID
    torch::Tensor value_loss = torch::binary_cross_entropy(value, value_target, {}, torch::Reduction::None);
#else
    torch::Tensor value_loss = torch::mse_loss(value, value_target, torch::Reduction::None);
#endif
#endif

    return { policy_loss, value_loss };
}

torch::Tensor LibTorchModel::contrastiveLoss(const std::vector<LearningData>& data) {
    auto [input, policy_target, value_target] = learningDataToTensor(data, device_, false);
    torch::Tensor representation = network_->encode(input);
    torch::Tensor loss = representation.norm();
    return loss;
}

std::vector<torch::Tensor> LibTorchModel::parameters() { return network_->parameters(); }
