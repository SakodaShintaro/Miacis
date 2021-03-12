#ifndef INFER_MODEL_HPP
#define INFER_MODEL_HPP

#include "neural_network.hpp"
#include <torch/script.h>

class InferModel {
public:
    InferModel() : device_(torch::kCPU) {}
    void load(const std::vector<std::string>& model_paths, int64_t gpu_id, int64_t opt_batch_size,
              const std::string& calibration_kifu_path, bool use_fp16);
    std::pair<std::vector<PolicyType>, std::vector<ValueType>> policyAndValueBatch(const std::vector<float>& inputs);
    std::tuple<torch::Tensor, torch::Tensor> infer(const std::vector<float>& inputs);
    std::pair<std::vector<PolicyType>, std::vector<ValueType>>
    decode(const std::tuple<torch::Tensor, torch::Tensor>& output) const;
    std::array<torch::Tensor, LOSS_TYPE_NUM> validLoss(const std::vector<LearningData>& data);

private:
    std::vector<torch::jit::Module> modules_;
    torch::Device device_;
    bool use_fp16_;
};

#endif