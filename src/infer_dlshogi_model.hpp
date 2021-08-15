#ifndef MIACIS_INFER_DLSHOGI_MODEL_HPP
#define MIACIS_INFER_DLSHOGI_MODEL_HPP

#ifdef DLSHOGI

#include "neural_network.hpp"
#include "search_options.hpp"
#include <torch/script.h>

class InferDLShogiModel {
public:
    InferDLShogiModel() : device_(torch::kCPU) {}
    void load(int64_t gpu_id, const SearchOptions& search_option);
    std::pair<std::vector<PolicyType>, std::vector<ValueType>> policyAndValueBatch(const std::vector<float>& inputs);
    std::tuple<torch::Tensor, torch::Tensor> infer(const std::vector<float>& inputs);
    std::array<torch::Tensor, LOSS_TYPE_NUM> validLoss(const std::vector<LearningData>& data);

private:
    torch::jit::Module network_;
    torch::Device device_;
    bool use_fp16_{};
};

#endif

#endif //MIACIS_INFER_DLSHOGI_MODEL_HPP