#include "learning_model.hpp"
#include "../common.hpp"
#include "../learn/learn.hpp"
#include "../shogi/move.hpp"
#include <torch/torch.h>

void LearningModel::load(const std::string& model_path, int64_t gpu_id) {
    module_ = torch::jit::load(model_path);
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    module_.to(device_);
}

void LearningModel::save(const std::string& model_path) { module_.save(model_path); }

std::array<torch::Tensor, LOSS_TYPE_NUM> LearningModel::loss(const std::vector<LearningData>& data) {
    torch::Tensor input = getInputTensor(data, device_);
    torch::Tensor policy_target = getPolicyTargetTensor(data, device_);
    torch::Tensor out = module_.forward({ input }).toTensor();
    auto list = torch::split(out, POLICY_DIM, 1);
    torch::Tensor policy = list[0];
    torch::Tensor value = list[1];

    torch::Tensor policy_logits = policy.view({ -1, POLICY_DIM });
    torch::Tensor policy_softmax = torch::softmax(policy_logits, 1);
    torch::Tensor focal_coeff = torch::pow(1 - policy_softmax, 2);
    torch::Tensor policy_loss = torch::sum(-policy_target * focal_coeff * torch::log_softmax(policy_logits, 1), 1, false);

#ifdef USE_CATEGORICAL
    torch::Tensor value_target = getCategoricalValueTargetTensor(data, device_);
    torch::Tensor value_loss = torch::sum(-value_target * torch::log_softmax(value, 1), 1, false);
#else
    torch::Tensor value_target = getValueTargetTensor(data, device_);
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
    torch::Tensor input = getInputTensor(data, device_);
    torch::Tensor policy_target = getPolicyTargetTensor(data, device_);
    torch::Tensor value_target = getValueTargetTensor(data, device_);
    torch::Tensor out = module_.forward({ input }).toTensor();
    auto list = torch::split(out, POLICY_DIM, 1);
    torch::Tensor policy_logit = list[0];
    torch::Tensor value_logit = list[1];

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

std::array<torch::Tensor, LOSS_TYPE_NUM> LearningModel::mixUpLoss(const std::vector<LearningData>& data, float alpha) {
    torch::Tensor input_tensor = getInputTensor(data, device_);
    torch::Tensor policy_target = getPolicyTargetTensor(data, device_);
    torch::Tensor value_target = getCategoricalValueTargetTensor(data, device_);

    //混合比率の振り出し
    std::gamma_distribution<float> gamma_dist(alpha);
    float gamma1 = gamma_dist(engine), gamma2 = gamma_dist(engine);
    float beta = gamma1 / (gamma1 + gamma2);

    //データのmixup
    input_tensor = beta * input_tensor + (1 - beta) * input_tensor.roll(1, 0);
    policy_target = beta * policy_target + (1 - beta) * policy_target.roll(1, 0);
    value_target = beta * value_target + (1 - beta) * value_target.roll(1, 0);

    torch::Tensor out = module_.forward({ input_tensor }).toTensor();
    auto list = torch::split(out, POLICY_DIM, 1);
    torch::Tensor policy = list[0];
    torch::Tensor value = list[1];

    torch::Tensor policy_logits = policy.view({ -1, POLICY_DIM });

    torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(policy_logits, 1), 1, false);

#ifdef USE_CATEGORICAL
    torch::Tensor value_loss = torch::sum(-value_target * torch::log_softmax(value, 1), 1, false);
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
    torch::Tensor input = getInputTensor(data, device_);
    torch::Tensor representation = module_.get_method("encode")({ input }).toTensor();
    torch::Tensor loss = representation.norm();
    return loss;
}

std::vector<torch::Tensor> LearningModel::getRepresentations(const std::vector<LearningData>& data) {
    torch::Tensor input = getInputTensor(data, device_);
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