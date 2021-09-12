#include "learning_model.hpp"
#include "../common.hpp"
#include "../include_switch.hpp"
#include "../learn/learn.hpp"
#include <torch/torch.h>

void LearningModel::load(const std::string& model_path, int64_t gpu_id) {
    module_ = torch::jit::load(model_path);
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    module_.to(device_);
}

void LearningModel::save(const std::string& model_path) { module_.save(model_path); }

std::array<torch::Tensor, LOSS_TYPE_NUM> LearningModel::loss(const std::vector<LearningData>& data) {
    auto [input, policy_target, value_target] = learningDataToTensor(data, device_);
    auto out = module_.forward({ input });
    auto tuple = out.toTuple();
    torch::Tensor policy = tuple->elements()[0].toTensor();
    torch::Tensor value = tuple->elements()[1].toTensor();

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

std::array<torch::Tensor, LOSS_TYPE_NUM> LearningModel::validLoss(const std::vector<LearningData>& data) {
#ifdef USE_CATEGORICAL
    auto [input, policy_target, value_target] = learningDataToTensor(data, device_);
    auto out = module_.forward({ input });
    auto tuple = out.toTuple();
    torch::Tensor policy_logit = tuple->elements()[0].toTensor();
    torch::Tensor value_logit = tuple->elements()[1].toTensor();

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

    //target側も数値に変換
    value_target = MIN_SCORE + (value_target + 0.5f) * VALUE_WIDTH;

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

std::array<torch::Tensor, LOSS_TYPE_NUM> LearningModel::mixUpLoss(const std::vector<LearningData>& data, float alpha) {
    auto [input_tensor, policy_target, value_target] = learningDataToTensor(data, device_);

    //混合比率の振り出し
    std::gamma_distribution<float> gamma_dist(alpha);
    float gamma1 = gamma_dist(engine), gamma2 = gamma_dist(engine);
    float beta = gamma1 / (gamma1 + gamma2);

    //データのmixup
    input_tensor = beta * input_tensor + (1 - beta) * input_tensor.roll(1, 0);
    policy_target = beta * policy_target + (1 - beta) * policy_target.roll(1, 0);
    value_target = beta * value_target + (1 - beta) * value_target.roll(1, 0);

    auto out = module_.forward({ input_tensor });
    auto tuple = out.toTuple();
    torch::Tensor policy = tuple->elements()[0].toTensor();
    torch::Tensor value = tuple->elements()[1].toTensor();

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

std::vector<torch::Tensor> LearningModel::parameters() {
    std::vector<torch::Tensor> parameters;
    for (auto p : module_.parameters()) {
        parameters.push_back(p);
    }
    return parameters;
}

torch::Tensor LearningModel::contrastiveLoss(const std::vector<LearningData>& data) {
    auto [input, policy_target, value_target] = learningDataToTensor(data, device_);
    torch::Tensor representation = module_.get_method("encode")({ input }).toTensor();
    torch::Tensor loss = representation.norm();
    return loss;
}

std::vector<torch::Tensor> LearningModel::getRepresentations(const std::vector<LearningData>& data) {
    auto [input, policy_target, value_target] = learningDataToTensor(data, device_);
    auto output = module_.get_method("get_representations")({ input });
    auto list = output.toTensorList();
    std::vector<torch::Tensor> result;
    for (auto t : list) {
        result.push_back(torch::Tensor(t));
    }
    return result;
    // module_->ge
    // return module_->getRepresentations(input);
}