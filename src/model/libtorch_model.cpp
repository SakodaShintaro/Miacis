#include "libtorch_model.hpp"
#include "../common.hpp"
#include "../include_switch.hpp"
#include "../learn/learn.hpp"

using namespace torch::nn;

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

Conv2DwithNormImpl::Conv2DwithNormImpl(int64_t input_ch, int64_t output_ch, int64_t kernel_size) {
    conv_ =
        register_module("conv_", Conv2d(Conv2dOptions(input_ch, output_ch, kernel_size).bias(false).padding(kernel_size / 2)));
    norm_ = register_module("norm_", LayerNorm(std::vector<int64_t>({ output_ch, BOARD_WIDTH, BOARD_WIDTH })));
}

torch::Tensor Conv2DwithNormImpl::forward(const torch::Tensor& x) {
    torch::Tensor t = x;
    t = conv_->forward(t);
    t = norm_->forward(t);
    return t;
}

ResidualBlockImpl::ResidualBlockImpl(int64_t channel_num, int64_t kernel_size, int64_t reduction) {
    conv_and_norm0_ = register_module("conv_and_norm0_", Conv2DwithNorm(channel_num, channel_num, kernel_size));
    conv_and_norm1_ = register_module("conv_and_norm1_", Conv2DwithNorm(channel_num, channel_num, kernel_size));
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

ValueHeadImpl::ValueHeadImpl(int64_t in_channels, int64_t out_dim, int64_t hidden_dim) {
    conv_and_norm_ = register_module("conv_and_norm_", Conv2DwithNorm(in_channels, in_channels, 1));
    linear0_ = register_module("linear0_", Linear(in_channels, hidden_dim));
    linear1_ = register_module("linear1_", Linear(hidden_dim, out_dim));
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

NetworkImpl::NetworkImpl() : blocks_(SHARE_BLOCK_NUM, nullptr) {
    first_conv_and_norm_ = register_module("first_conv_and_norm_", Conv2DwithNorm(INPUT_CHANNEL_NUM, CHANNEL_NUM, KERNEL_SIZE));
    for (int32_t i = 0; i < SHARE_BLOCK_NUM; i++) {
        blocks_[i] = register_module("blocks_" + std::to_string(i), ResidualBlock(CHANNEL_NUM, KERNEL_SIZE, REDUCTION));
    }
    policy_head_ =
        register_module("policy_head_", Conv2d(Conv2dOptions(CHANNEL_NUM, POLICY_CHANNEL_NUM, 1).padding(0).bias(true)));
    value_head_ = register_module("value_head_", ValueHead(CHANNEL_NUM, BIN_SIZE, VALUE_HIDDEN_NUM));
    ponder_head_ = register_module("ponder_head_", ValueHead(CHANNEL_NUM, 1, VALUE_HIDDEN_NUM));
}

torch::Tensor NetworkImpl::firstEncode(const torch::Tensor& x) {
    torch::Tensor r = x;
    r = first_conv_and_norm_->forward(r);
    r = activation(r);
    return r;
}

torch::Tensor NetworkImpl::applyOneLoop(const torch::Tensor& x) {
    torch::Tensor r = x;
    for (ResidualBlock& block : blocks_) {
        r = block->forward(r);
    }
    return r;
}

torch::Tensor NetworkImpl::encode(const torch::Tensor& x, int64_t loop_num) {
    torch::Tensor r = x;
    r = first_conv_and_norm_->forward(r);
    r = activation(r);

    for (int64_t _ = 0; _ < loop_num; _++) {
        r = applyOneLoop(r);
    }

    return r;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> NetworkImpl::decode(const torch::Tensor& representation) {
    //policy
    torch::Tensor policy = policy_head_->forward(representation).view({ -1, POLICY_DIM });

    //value
    torch::Tensor value = value_head_->forward(representation);

#ifndef USE_CATEGORICAL
#ifdef USE_SIGMOID
    value = torch::sigmoid(value);
#else
    value = torch::tanh(value);
#endif
#endif

    //ponder(halt probability)
    torch::Tensor ponder = ponder_head_->forward(representation);
    ponder = torch::sigmoid(ponder).flatten();

    return { policy, value, ponder };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> NetworkImpl::forward(const torch::Tensor& x, int64_t loop_num) {
    return decode(encode(x, loop_num));
}

void LibTorchModel::load(const std::string& model_path, int64_t gpu_id) {
    torch::load(network_, model_path);
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    network_->to(device_);
}

void LibTorchModel::save(const std::string& model_path) { torch::save(network_, model_path); }

//std::array<torch::Tensor, LOSS_TYPE_NUM> LibTorchModel::loss(const std::vector<LearningData>& data) {
//    auto [input, policy_target, value_target] = learningDataToTensor(data, device_, false);
//    auto out = network_->forward(input);
//    torch::Tensor x = network_->firstEncode(input);
//
//    const int64_t batch_size = data.size();
//    torch::Tensor policy_loss_sum = torch::zeros({ batch_size }).to(device_);
//    torch::Tensor value_loss_sum = torch::zeros({ batch_size }).to(device_);
//    torch::Tensor ponder_loss_sum = torch::zeros({ batch_size }).to(device_);
//    torch::Tensor remaining_prob = torch::ones({ batch_size }).to(device_);
//    torch::Tensor target_remaining_prob = torch::ones({ batch_size }).to(device_);
//    constexpr int64_t BASE_LOOP_NUM = BLOCK_NUM / SHARE_BLOCK_NUM;
//    constexpr float TARGET_CONSTANT_PROB = (1.0 / BASE_LOOP_NUM);
//
//    for (int64_t loop_num = 0; loop_num < 2 * BLOCK_NUM / SHARE_BLOCK_NUM; loop_num++) {
//        x = network_->applyOneLoop(x);
//        auto [policy, value, ponder] = network_->decode(x);
//
//        //今回で初めて推論が止まる確率 = (まだ推論が止まっていない確率) * (今回で止めると決断する確率)
//        torch::Tensor halt_prob = remaining_prob * ponder;
//        torch::Tensor target_halt_prob = target_remaining_prob * TARGET_CONSTANT_PROB;
//
//        torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(policy, 1), 1, false);
//        policy_loss_sum = policy_loss_sum + policy_loss * halt_prob;
//
//#ifdef USE_CATEGORICAL
//        torch::Tensor value_loss = torch::nll_loss(torch::log_softmax(value, 1), value_target);
//#else
//        value = value.view(-1);
//#ifdef USE_SIGMOID
//        torch::Tensor value_loss = torch::binary_cross_entropy(value, value_target, {}, torch::Reduction::None);
//#else
//        torch::Tensor value_loss = torch::mse_loss(value, value_target, torch::Reduction::None);
//#endif
//#endif
//        value_loss_sum = value_loss_sum + value_loss * halt_prob;
//
//        ponder_loss_sum = ponder_loss_sum + torch::binary_cross_entropy(halt_prob, target_halt_prob);
//
//        remaining_prob = remaining_prob * (1 - ponder);
//        target_halt_prob = target_halt_prob * (1 - TARGET_CONSTANT_PROB);
//    }
//
//    return { policy_loss_sum, value_loss_sum };
//}

std::array<torch::Tensor, LOSS_TYPE_NUM> LibTorchModel::loss(const std::vector<LearningData>& data) {
    auto [input, policy_target, value_target] = learningDataToTensor(data, device_, false);
    auto [policy, value, _] = network_->forward(input);

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

std::array<torch::Tensor, LOSS_TYPE_NUM> LibTorchModel::validLoss(const std::vector<LearningData>& data, int64_t loop_num) {
#ifdef USE_CATEGORICAL
    auto [input, policy_target, value_target] = learningDataToTensor(data, device_, true);
    auto [policy_logit, value_logit, ponder] = network_->forward(input, loop_num);

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
    auto [input, policy_target, value_target] = learningDataToTensor(data, device_, false);

    //混合比率の振り出し
    std::gamma_distribution<float> gamma_dist(alpha);
    float gamma1 = gamma_dist(engine), gamma2 = gamma_dist(engine);
    float beta = gamma1 / (gamma1 + gamma2);

    //データのmixup
    input = beta * input + (1 - beta) * input.roll(1, 0);
    policy_target = beta * policy_target + (1 - beta) * policy_target.roll(1, 0);
    value_target = beta * value_target + (1 - beta) * value_target.roll(1, 0);

    auto [policy_logit, value, ponder] = network_->forward(input);

    torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(policy_logit, 1), 1, false);

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
