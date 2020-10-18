#include "base_model.hpp"

BaseModel::BaseModel(const SearchOptions& options)
    : search_options_(options), device_(torch::kCUDA), fp16_(false), freeze_encoder_(true) {
    encoder_ = register_module("encoder_", StateEncoder());
    sim_policy_head_ = register_module("sim_policy_head_", torch::nn::Linear(StateEncoderImpl::HIDDEN_DIM, POLICY_DIM));
}

void BaseModel::setGPU(int16_t gpu_id, bool fp16) {
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    fp16_ = fp16;
    (fp16_ ? to(device_, torch::kHalf) : to(device_, torch::kFloat));
}

void BaseModel::loadPretrain(const std::string& encoder_path, const std::string& policy_head_path) {
    std::ifstream encoder_file(encoder_path);
    if (encoder_file.is_open()) {
        torch::load(encoder_, encoder_path);
    }
    std::ifstream policy_head_file(policy_head_path);
    if (policy_head_file.is_open()) {
        torch::load(sim_policy_head_, policy_head_path);
    }
}

void BaseModel::setOption(bool freeze_encoder) { freeze_encoder_ = freeze_encoder; }