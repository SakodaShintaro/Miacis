#ifndef INFER_MODEL_HPP
#define INFER_MODEL_HPP

#include "neural_network.hpp"
#include <torch/script.h>

class InferModel {
public:
    InferModel() : device_(torch::kCPU) {}
    void load(const std::string& model_path, int64_t gpu_id, int64_t opt_batch_size, const std::string& calibration_kifu_path);
    std::pair<std::vector<PolicyType>, std::vector<ValueType>> policyAndValueBatch(const std::vector<float>& inputs);
    std::array<torch::Tensor, LOSS_TYPE_NUM> validLoss(const std::vector<LearningData>& data);

private:
    torch::jit::Module module_;
    torch::Device device_;
};

#endif