#include "infer_dlshogi_model.hpp"

#ifdef DLSHOGI

#include "../common.hpp"
#include "../include_switch.hpp"
#include "../learn.hpp"
#include "dataset.hpp"
#include "infer_model.hpp"
#include <torch/torch.h>
#include <trtorch/ptq.h>
#include <trtorch/trtorch.h>

void InferDLShogiModel::load(int64_t gpu_id, const SearchOptions& search_option) {
    //マルチGPU環境で同時にloadすると時々Segmentation Faultが発生するので排他制御を入れる
    static std::mutex load_mutex;
    std::lock_guard<std::mutex> guard(load_mutex);

    torch::jit::Module module = torch::jit::load(search_option.model_name);
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    module.to(device_);
    module.eval();

    const int64_t opt_batch_size = search_option.search_batch_size;
    std::vector<int64_t> in1_min = { 1, DLSHOGI_FEATURES1_NUM, BOARD_WIDTH, BOARD_WIDTH };
    std::vector<int64_t> in1_opt = { opt_batch_size, DLSHOGI_FEATURES1_NUM, BOARD_WIDTH, BOARD_WIDTH };
    std::vector<int64_t> in1_max = { opt_batch_size * 2, DLSHOGI_FEATURES1_NUM, BOARD_WIDTH, BOARD_WIDTH };
    std::vector<int64_t> in2_min = { 1, DLSHOGI_FEATURES2_NUM, BOARD_WIDTH, BOARD_WIDTH };
    std::vector<int64_t> in2_opt = { opt_batch_size, DLSHOGI_FEATURES2_NUM, BOARD_WIDTH, BOARD_WIDTH };
    std::vector<int64_t> in2_max = { opt_batch_size * 2, DLSHOGI_FEATURES2_NUM, BOARD_WIDTH, BOARD_WIDTH };

    use_fp16_ = search_option.use_fp16;
    trtorch::CompileSpec::InputRange range1(in1_min, in1_opt, in1_max);
    trtorch::CompileSpec::InputRange range2(in2_min, in2_opt, in2_max);
    trtorch::CompileSpec info({ range1, range2 });

    if (use_fp16_) {
        info.op_precision = torch::kHalf;
        info.device.gpu_id = gpu_id;
        network_ = trtorch::CompileGraph(module, info);
    } else {
        using namespace torch::data;
        const bool use_calibration_cache = search_option.use_calibration_cache;
        auto raw_dataset = (use_calibration_cache ? CalibrationDataset()
                                                  : CalibrationDataset(search_option.calibration_kifu_path, opt_batch_size * 2));
        auto dataset = raw_dataset.map(transforms::Stack<>());
        auto dataloader = make_data_loader(std::move(dataset), DataLoaderOptions().batch_size(opt_batch_size).workers(1));

        auto calibrator = trtorch::ptq::make_int8_calibrator<nvinfer1::IInt8MinMaxCalibrator>(
            std::move(dataloader), search_option.calibration_cache_path, use_calibration_cache);

        info.op_precision = torch::kI8;
        info.device.gpu_id = gpu_id;
        info.ptq_calibrator = calibrator;
        info.workspace_size = (1ull << 29);
        info.max_batch_size = opt_batch_size * 2;

        network_ = trtorch::CompileGraph(module, info);
    }
}

std::pair<std::vector<PolicyType>, std::vector<ValueType>>
InferDLShogiModel::policyAndValueBatch(const std::vector<float>& inputs) {
    return tensorToVector(infer(inputs));
}

std::tuple<torch::Tensor, torch::Tensor> InferDLShogiModel::infer(const std::vector<float>& inputs) {
    torch::Tensor x = inputVectorToTensor(inputs).to(device_);
    if (use_fp16_) {
        x = x.to(torch::kFloat16);
    }
    std::vector<torch::Tensor> xs = x.split(DLSHOGI_FEATURES1_NUM, 1);
    torch::Tensor x1 = xs[0];
    torch::Tensor x2 = xs[1];

    auto out = network_.forward({ x1, x2 });
    auto tuple = out.toTuple();
    torch::Tensor policy = tuple->elements()[0].toTensor();
    torch::Tensor value = tuple->elements()[1].toTensor();

    //DLShogiはsigmoidを使っているのでtanhに合わせて引き伸ばす
    value = value * 2 - 1;

    //CPUに持ってくる
    policy = policy.cpu();

    //valueはcategoricalのときだけはsoftmaxをかけてからcpuへ
#ifdef USE_CATEGORICAL
    std::cout << "dlshogiモデルはCategoricalモードに対応していない" << std::endl;
    std::exit(1);
#else
    value = value.cpu();
#endif

    return std::make_tuple(policy, value);
}

std::array<torch::Tensor, LOSS_TYPE_NUM> InferDLShogiModel::validLoss(const std::vector<LearningData>& data) {
#ifdef USE_CATEGORICAL
    std::cout << "dlshogiモデルはCategoricalモードに対応していない" << std::endl;
    std::exit(1);
#else
    auto [input, policy_target, value_target] = learningDataToTensor(data, device_, true);

    if (use_fp16_) {
        input = input.to(torch::kFloat16);
    }

    std::vector<torch::Tensor> xs = input.split(DLSHOGI_FEATURES1_NUM, 1);
    torch::Tensor x1 = xs[0];
    torch::Tensor x2 = xs[1];

    auto out = network_.forward({ x1, x2 });
    auto tuple = out.toTuple();
    torch::Tensor policy_logits = tuple->elements()[0].toTensor();
    torch::Tensor value = tuple->elements()[1].toTensor();

    torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(policy_logits, 1), 1, false);

    value = value.view(-1);
#ifdef USE_SIGMOID
    torch::Tensor value_loss = torch::binary_cross_entropy(value, value_target, {}, torch::Reduction::None);
#else
    value = value * 2 - 1;
    torch::Tensor value_loss = torch::mse_loss(value, value_target, torch::Reduction::None);
#endif

    return { policy_loss, value_loss };
#endif
}

#endif