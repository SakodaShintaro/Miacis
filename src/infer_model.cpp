#include "infer_model.hpp"
#include "common.hpp"
#include "dataset.hpp"
#include "include_switch.hpp"
#include <torch/torch.h>
#include <trtorch/ptq.h>
#include <trtorch/trtorch.h>

void InferModel::load(const std::string& model_path, int64_t gpu_id, int64_t opt_batch_size) {
    torch::jit::Module module = torch::jit::load(model_path);
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    module.to(device_);
    module.eval();

    std::vector<int64_t> in_min = { 1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH };
    std::vector<int64_t> in_opt = { opt_batch_size, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH };
    std::vector<int64_t> in_max = { opt_batch_size * 2, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH };

    auto dataset = CalibrationDataset("/home/sakoda/data/floodgate_kifu/valid").map(torch::data::transforms::Stack<>());
    auto dataloader =
        torch::data::make_data_loader(std::move(dataset), torch::data::DataLoaderOptions().batch_size(128).workers(1));

    const std::string name = "calibration_cache_file.txt";
    auto calibrator = trtorch::ptq::make_int8_calibrator<nvinfer1::IInt8MinMaxCalibrator>(std::move(dataloader), name, true);

    //trtorch
    trtorch::CompileSpec::InputRange range(in_min, in_opt, in_max);
    trtorch::CompileSpec info({ range });
    info.op_precision = torch::kI8;
    info.device.gpu_id = gpu_id;
    info.ptq_calibrator = calibrator;
    info.workspace_size = (1ull << 29);
    info.max_batch_size = opt_batch_size * 2;

    module_ = trtorch::CompileGraph(module, info);
}

std::pair<std::vector<PolicyType>, std::vector<ValueType>> InferModel::policyAndValueBatch(const std::vector<float>& inputs) {
    torch::Tensor x = torch::tensor(inputs).to(device_);
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
    float* p = policy.data_ptr<float>();
    for (uint64_t i = 0; i < batch_size; i++) {
        policies[i].assign(p + i * POLICY_DIM, p + (i + 1) * POLICY_DIM);
    }

#ifdef USE_CATEGORICAL
    value = torch::softmax(value, 1).cpu();
    float* value_p = value.data_ptr<float>();
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