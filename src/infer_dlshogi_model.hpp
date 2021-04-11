#ifndef MIACIS_INFER_DLSHOGI_MODEL_HPP
#define MIACIS_INFER_DLSHOGI_MODEL_HPP

#ifdef SHOGI

#include "neural_network.hpp"
#include <torch/script.h>

class InferDLShogiModel {
public:
    InferDLShogiModel() : device_(torch::kCPU) {}
    void load(const std::string& model_path, int64_t gpu_id, int64_t opt_batch_size, const std::string& calibration_kifu_path,
              bool use_fp16);
    std::pair<std::vector<PolicyType>, std::vector<ValueType>> policyAndValueBatch(const std::vector<float>& inputs);
    std::tuple<torch::Tensor, torch::Tensor> infer(const std::vector<float>& inputs);
    std::array<torch::Tensor, LOSS_TYPE_NUM> validLoss(const std::vector<LearningData>& data);

private:
    torch::jit::Module module_;
    torch::Device device_;
    bool use_fp16_{};
};

#endif

#endif //MIACIS_INFER_DLSHOGI_MODEL_HPP