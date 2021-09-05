#include "infer_trtorch_model.hpp"
#include "../common.hpp"
#include "../include_switch.hpp"
#include "../learn/learn.hpp"
#include "dataset.hpp"
#include <torch/torch.h>
#include <trtorch/ptq.h>
#include <trtorch/trtorch.h>

#ifndef DLSHOGI
void InferTRTorchModel::load(int64_t gpu_id, const SearchOptions& search_option) {
    //マルチGPU環境で同時にloadすると時々Segmentation Faultが発生するので排他制御を入れる
    static std::mutex load_mutex;
    std::lock_guard<std::mutex> guard(load_mutex);

    torch::jit::Module module = torch::jit::load(search_option.model_name);
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    module.to(device_);
    module.eval();

    const int64_t opt_batch_size = search_option.search_batch_size;
    std::vector<int64_t> in_min = { 1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH };
    std::vector<int64_t> in_opt = { opt_batch_size, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH };
    std::vector<int64_t> in_max = { opt_batch_size * 2, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH };

    use_fp16_ = search_option.use_fp16;
    if (use_fp16_) {
        trtorch::CompileSpec::InputRange range(in_min, in_opt, in_max);
        trtorch::CompileSpec info({ range });
        info.op_precision = torch::kHalf;
        info.device.gpu_id = gpu_id;
        module_ = trtorch::CompileGraph(module, info);
    } else {
        using namespace torch::data;
        const bool use_calibration_cache = search_option.use_calibration_cache;
        auto raw_dataset = (use_calibration_cache ? CalibrationDataset()
                                                  : CalibrationDataset(search_option.calibration_kifu_path, opt_batch_size * 2));
        auto dataset = raw_dataset.map(transforms::Stack<>());
        auto dataloader = make_data_loader(std::move(dataset), DataLoaderOptions().batch_size(opt_batch_size).workers(1));

        auto calibrator = trtorch::ptq::make_int8_calibrator<nvinfer1::IInt8MinMaxCalibrator>(
            std::move(dataloader), search_option.calibration_cache_path, use_calibration_cache);

        trtorch::CompileSpec::InputRange range(in_min, in_opt, in_max);
        trtorch::CompileSpec info({ range });
        info.op_precision = torch::kI8;
        info.device.gpu_id = gpu_id;
        info.ptq_calibrator = calibrator;
        info.workspace_size = (1ull << 29);
        info.max_batch_size = opt_batch_size * 2;

        module_ = trtorch::CompileGraph(module, info);
    }
}

std::pair<std::vector<PolicyType>, std::vector<ValueType>>
InferTRTorchModel::policyAndValueBatch(const std::vector<float>& inputs) {
    return tensorToVector(infer(inputs));
}

std::tuple<torch::Tensor, torch::Tensor> InferTRTorchModel::infer(const std::vector<float>& inputs) {
    torch::Tensor x = inputVectorToTensor(inputs).to(device_);
    if (use_fp16_) {
        x = x.to(torch::kFloat16);
    }
    auto out = module_.forward({ x });
    auto tuple = out.toTuple();
    torch::Tensor policy = tuple->elements()[0].toTensor();
    torch::Tensor value = tuple->elements()[1].toTensor();

    //CPUに持ってくる
    policy = policy.cpu();

    //valueはcategoricalのときだけはsoftmaxをかけてからcpuへ
#ifdef USE_CATEGORICAL
    value = torch::softmax(value, 1).cpu();
#else
    value = value.cpu();
#endif

    return std::make_tuple(policy, value);
}

std::array<torch::Tensor, LOSS_TYPE_NUM> InferTRTorchModel::validLoss(const std::vector<LearningData>& data) {
    auto [input, policy_target, value_target] = learningDataToTensor(data, device_);
    if (use_fp16_) {
        input = input.to(torch::kFloat16);
    }
    auto out = module_.forward({ input });
    auto tuple = out.toTuple();
    torch::Tensor policy_logit = tuple->elements()[0].toTensor().view({ -1, POLICY_DIM });
    torch::Tensor value = tuple->elements()[1].toTensor();

    torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(policy_logit, 1), 1, false);

#ifdef USE_CATEGORICAL
    //Valueの分布を取得
    torch::Tensor value_cat = torch::softmax(value, 1);

    //i番目の要素が示す値はMIN_SCORE + (i + 0.5) * VALUE_WIDTH
    std::vector<float> each_value;
    for (int64_t i = 0; i < BIN_SIZE; i++) {
        each_value.emplace_back(MIN_SCORE + (i + 0.5) * VALUE_WIDTH);
    }
    torch::Tensor each_value_tensor = torch::tensor(each_value).to(device_);

    //Categorical分布と内積を取ることで期待値を求める
    value = (each_value_tensor * value_cat).sum(1);

    //target側も数値に変換
    value_target = MIN_SCORE + (value_target + 0.5f) * VALUE_WIDTH;

#else //Scalarモデルの場合
    value = value.view(-1);
#endif

    //Sigmoidのときはbce, tanhのときはmse
#ifdef USE_SIGMOID
    torch::Tensor value_loss = torch::binary_cross_entropy(value, value_target, {}, torch::Reduction::None);
#else
    torch::Tensor value_loss = torch::mse_loss(value, value_target, torch::Reduction::None);
#endif

    return { policy_loss, value_loss };
}

#endif