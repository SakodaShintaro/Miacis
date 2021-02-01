#ifndef INFER_MODEL_HPP
#define INFER_MODEL_HPP

#include "neural_network.hpp"
#include <torch/script.h>

class InferModel {
public:
    InferModel() : device_(torch::kCPU) {}
    void load(const std::string& model_path, int64_t gpu_id);
    std::pair<std::vector<PolicyType>, std::vector<ValueType>> policyAndValueBatch(const std::vector<float>& inputs);

private:
    torch::jit::Module module_;
    torch::Device device_;
};

#endif