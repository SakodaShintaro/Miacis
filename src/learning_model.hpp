#ifndef LEARNING_MODEL_HPP
#define LEARNING_MODEL_HPP

#include "neural_network.hpp"
#include <torch/script.h>

class LearningModel {
public:
    LearningModel() : device_(torch::kCPU) {}
    void load(const std::string& model_path, int64_t gpu_id);
    void save(const std::string& model_path);
    torch::Tensor encode(const std::vector<float>& inputs) const;
    std::array<torch::Tensor, LOSS_TYPE_NUM> loss(const std::vector<LearningData>& data);
    std::array<torch::Tensor, LOSS_TYPE_NUM> validLoss(const std::vector<LearningData>& data);
    std::vector<torch::Tensor> parameters();

    void train() { module_.train() ;}
    void eval() { module_.eval(); }

private:
    torch::jit::Module module_;
    torch::Device device_;
};

#endif