#include "common.hpp"
#include "../include_switch.hpp"

torch::Tensor getPolicyTeacher(const std::vector<LearningData>& data, torch::Device device) {
    const int64_t batch_size = data.size();
    std::vector<float> policy_teachers(batch_size * POLICY_DIM, 0.0);
    for (int64_t i = 0; i < batch_size; i++) {
        for (const std::pair<int32_t, float>& e : data[i].policy) {
            policy_teachers[i * POLICY_DIM + e.first] = e.second;
        }
    }
    return torch::tensor(policy_teachers).to(device).view({ batch_size, POLICY_DIM });
}