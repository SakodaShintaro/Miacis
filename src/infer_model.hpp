#ifndef INFER_MODEL_HPP
#define INFER_MODEL_HPP

#include "neural_network.hpp"
#include "search_options.hpp"
#include <torch/script.h>

class InferModel {
public:
    InferModel() : device_(torch::kCPU) {}
    void load(int64_t gpu_id, bool use_calibration_cache, const SearchOptions& search_option);

    std::pair<std::vector<PolicyType>, std::vector<ValueType>> policyAndValueBatch(const std::vector<float>& inputs);
    std::tuple<torch::Tensor, torch::Tensor> infer(const std::vector<float>& inputs);
    std::array<torch::Tensor, LOSS_TYPE_NUM> validLoss(const std::vector<LearningData>& data);

private:
    void load(const std::string& model_path, int64_t gpu_id, int64_t opt_batch_size, bool use_calibration_cache,
              const std::string& calibration_kifu_path, const std::string& calibration_cache_path, bool use_fp16);

    torch::jit::Module module_;
    torch::Device device_;
    bool use_fp16_{};
};

#endif