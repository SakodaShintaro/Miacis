#ifndef INFER_MODEL_HPP
#define INFER_MODEL_HPP

#include "neural_network.hpp"
#include "search_options.hpp"
#include <torch/script.h>

#ifdef DLSHOGI

#include "infer_dlshogi_model.hpp"
#include "infer_dlshogi_onnx_model.hpp"
using InferModel = InferDLShogiOnnxModel;

#else

class InferModel {
public:
    InferModel() : device_(torch::kCPU) {}
    void load(int64_t gpu_id, const SearchOptions& search_option);

    std::pair<std::vector<PolicyType>, std::vector<ValueType>> policyAndValueBatch(const std::vector<float>& inputs);
    std::tuple<torch::Tensor, torch::Tensor> infer(const std::vector<float>& inputs);
    std::array<torch::Tensor, LOSS_TYPE_NUM> validLoss(const std::vector<LearningData>& data);

private:
    torch::jit::Module module_;
    torch::Device device_;
    bool use_fp16_{};
};

#endif

#endif