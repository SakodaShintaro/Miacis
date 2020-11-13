#include "common.hpp"
#include "../include_switch.hpp"

torch::Tensor policyLoss(const torch::Tensor& policy_logit, const torch::Tensor& policy_teacher) {
    //policy_logit, policy_teacherのshapeは(batch_size, policy_dim)であるとする

    //policyが尖っていると損失値が非常に大きくなってしまうので適当な値で制限をかける
    static const float LOG_SOFTMAX_THRESHOLD = -20;

    torch::Tensor log_softmax = torch::log_softmax(policy_logit, 1);
    torch::Tensor clipped = torch::clamp_min(log_softmax, LOG_SOFTMAX_THRESHOLD);
    return -(policy_teacher * clipped).sum(1).mean(0);
}

torch::Tensor entropyLoss(const torch::Tensor& policy_logit) {
    return -policyLoss(policy_logit, torch::softmax(policy_logit, 1));
}