#ifndef MIACIS_SEARCH_NN_COMMON_HPP
#define MIACIS_SEARCH_NN_COMMON_HPP

#include "../neural_network.hpp"

torch::Tensor getPolicyTeacher(const std::vector<LearningData>& data, torch::Device device);
torch::Tensor policyLoss(const torch::Tensor& policy_logit, const torch::Tensor& policy_teacher);
torch::Tensor entropyLoss(const torch::Tensor& policy_logit);

#endif //MIACIS_SEARCH_NN_COMMON_HPP