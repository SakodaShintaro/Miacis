#include "infer_model.hpp"
#include "common.hpp"
#include "include_switch.hpp"
#include <torch/torch.h>
#include <trtorch/trtorch.h>

void InferModel::load(const std::string& model_path, int64_t gpu_id) {
    module_ = torch::jit::load(model_path);
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    module_.to(torch::kCUDA, torch::kHalf);
    module_.eval();

    torch::Tensor in_min = torch::randn({ 1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH }, torch::kCUDA).to(torch::kHalf);
    torch::Tensor in_opt = torch::randn({ 128, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH }, torch::kCUDA).to(torch::kHalf);
    torch::Tensor in_max = torch::randn({ 256, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH }, torch::kCUDA).to(torch::kHalf);

    //trtorch
    trtorch::CompileSpec::InputRange range(in_min.sizes(), in_opt.sizes(), in_max.sizes());
    trtorch::CompileSpec info({ range });
    info.op_precision = torch::kHalf;
    module_ = trtorch::CompileGraph(module_, info);
}

std::pair<std::vector<PolicyType>, std::vector<ValueType>> InferModel::policyAndValueBatch(const std::vector<float>& inputs) {
    torch::Tensor x = torch::tensor(inputs).to(device_).to(torch::kHalf);
    x = x.view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
    auto out = module_.forward({ x });
    auto tuple = out.toTuple();
    torch::Tensor policy = tuple->elements()[0].toTensor();
    torch::Tensor value = tuple->elements()[1].toTensor();

    uint64_t batch_size = inputs.size() / (SQUARE_NUM * INPUT_CHANNEL_NUM);

    std::vector<PolicyType> policies(batch_size);
    std::vector<ValueType> values(batch_size);

    //CPUに持ってくる
    policy = policy.cpu();
    torch::Half* p = policy.data_ptr<torch::Half>();
    for (uint64_t i = 0; i < batch_size; i++) {
        policies[i].assign(p + i * POLICY_DIM, p + (i + 1) * POLICY_DIM);
    }

#ifdef USE_CATEGORICAL
    value = torch::softmax(value, 1).cpu();
    torch::Half* value_p = value.data_ptr<torch::Half>();
    for (uint64_t i = 0; i < batch_size; i++) {
        std::copy(value_p + i * BIN_SIZE, value_p + (i + 1) * BIN_SIZE, values[i].begin());
    }
#else
    //CPUに持ってくる
    value = value.cpu();
    std::copy(value.data_ptr<torch::Half>(), value.data_ptr<torch::Half>() + batch_size, values.begin());
#endif
    return std::make_pair(policies, values);
}