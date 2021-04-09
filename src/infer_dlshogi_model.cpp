#include "infer_dlshogi_model.hpp"

#include "common.hpp"
#include "dataset.hpp"
#include "include_switch.hpp"
#include "infer_model.hpp"
#include <torch/torch.h>
#include <trtorch/ptq.h>
#include <trtorch/trtorch.h>

void InferDLShogiModel::load(const std::string& model_path, int64_t gpu_id, int64_t opt_batch_size,
                             const std::string& calibration_kifu_path, bool use_fp16) {
    //マルチGPU環境で同時にloadすると時々Segmentation Faultが発生するので排他制御を入れる
    static std::mutex load_mutex;
    std::lock_guard<std::mutex> guard(load_mutex);

    torch::jit::Module module = torch::jit::load(model_path);
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    module.to(device_);
    module.eval();

    std::vector<int64_t> in1_min = { 1, DLSHOGI_FEATURES1_NUM, BOARD_WIDTH, BOARD_WIDTH };
    std::vector<int64_t> in1_opt = { opt_batch_size, DLSHOGI_FEATURES1_NUM, BOARD_WIDTH, BOARD_WIDTH };
    std::vector<int64_t> in1_max = { opt_batch_size * 2, DLSHOGI_FEATURES1_NUM, BOARD_WIDTH, BOARD_WIDTH };
    std::vector<int64_t> in2_min = { 1, DLSHOGI_FEATURES2_NUM, BOARD_WIDTH, BOARD_WIDTH };
    std::vector<int64_t> in2_opt = { opt_batch_size, DLSHOGI_FEATURES2_NUM, BOARD_WIDTH, BOARD_WIDTH };
    std::vector<int64_t> in2_max = { opt_batch_size * 2, DLSHOGI_FEATURES2_NUM, BOARD_WIDTH, BOARD_WIDTH };

    use_fp16_ = use_fp16;
    trtorch::CompileSpec::InputRange range1(in1_min, in1_opt, in1_max);
    trtorch::CompileSpec::InputRange range2(in2_min, in2_opt, in2_max);
    trtorch::CompileSpec info({ range1, range2 });

    if (use_fp16_) {
        info.op_precision = torch::kHalf;
        info.device.gpu_id = gpu_id;
        module_ = trtorch::CompileGraph(module, info);
    } else {
        using namespace torch::data;
        auto dataset = CalibrationDataset(calibration_kifu_path, opt_batch_size * 2).map(transforms::Stack<>());
        auto dataloader = make_data_loader(std::move(dataset), DataLoaderOptions().batch_size(opt_batch_size).workers(1));

        const std::string name = "calibration_cache_dlshogi.txt";
        auto calibrator = trtorch::ptq::make_int8_calibrator<nvinfer1::IInt8MinMaxCalibrator>(std::move(dataloader), name, false);

        info.op_precision = torch::kI8;
        info.device.gpu_id = gpu_id;
        info.ptq_calibrator = calibrator;
        info.workspace_size = (1ull << 29);
        info.max_batch_size = opt_batch_size * 2;

        module_ = trtorch::CompileGraph(module, info);
    }
}

std::pair<std::vector<PolicyType>, std::vector<ValueType>>
InferDLShogiModel::policyAndValueBatch(const std::vector<float>& inputs) {
    return decode(infer(inputs));
}

std::tuple<torch::Tensor, torch::Tensor> InferDLShogiModel::infer(const std::vector<float>& inputs) {
    torch::Tensor x = torch::tensor(inputs).to(device_);
    x = x.view({ -1, (DLSHOGI_FEATURES1_NUM + DLSHOGI_FEATURES2_NUM), BOARD_WIDTH, BOARD_WIDTH });

    if (use_fp16_) {
        x = x.to(torch::kFloat16);
    }

    std::vector<torch::Tensor> xs = x.split(DLSHOGI_FEATURES1_NUM, 1);
    torch::Tensor x1 = xs[0];
    torch::Tensor x2 = xs[1];

    auto out = module_.forward({ x1, x2 });
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

std::pair<std::vector<PolicyType>, std::vector<ValueType>>
InferDLShogiModel::decode(const std::tuple<torch::Tensor, torch::Tensor>& output) const {
    const auto& [policy, value] = output;
    uint64_t batch_size = policy.size(0);

    std::vector<PolicyType> policies(batch_size);
    std::vector<ValueType> values(batch_size);

    if (use_fp16_) {
        torch::Half* p = policy.data_ptr<torch::Half>();
        for (uint64_t i = 0; i < batch_size; i++) {
            policies[i].assign(p + i * POLICY_DIM, p + (i + 1) * POLICY_DIM);
        }
    } else {
        float* p = policy.data_ptr<float>();
        for (uint64_t i = 0; i < batch_size; i++) {
            policies[i].assign(p + i * POLICY_DIM, p + (i + 1) * POLICY_DIM);
        }
    }

#ifdef USE_CATEGORICAL
    std::cout << "dlshogiモデルはCategoricalモードに対応していない" << std::endl;
    std::exit(1);
#else
    std::copy(value.data_ptr<float>(), value.data_ptr<float>() + batch_size, values.begin());
#endif
    return std::make_pair(policies, values);
}

std::array<torch::Tensor, LOSS_TYPE_NUM> InferDLShogiModel::validLoss(const std::vector<LearningData>& data) {
#ifdef USE_CATEGORICAL
    std::cout << "dlshogiモデルはCategoricalモードに対応していない" << std::endl;
    std::exit(1);
#else
    static Position pos;
    std::vector<float> inputs;
    std::vector<float> policy_teachers(data.size() * POLICY_DIM, 0.0);
    std::vector<ValueTeacherType> value_teachers;

    for (uint64_t i = 0; i < data.size(); i++) {
        pos.fromStr(data[i].position_str);

        //入力
        const std::vector<float> feature = pos.makeDLShogiFeature();
        inputs.insert(inputs.end(), feature.begin(), feature.end());

        //policyの教師信号
        for (const std::pair<int32_t, float>& e : data[i].policy) {
            policy_teachers[i * POLICY_DIM + e.first] = e.second;
        }

        //valueの教師信号
        value_teachers.push_back(data[i].value);
    }

    torch::Tensor x = torch::tensor(inputs).to(device_);
    x = x.view({ -1, (DLSHOGI_FEATURES1_NUM + DLSHOGI_FEATURES2_NUM), BOARD_WIDTH, BOARD_WIDTH });
    if (use_fp16_) {
        x = x.to(torch::kFloat16);
    }

    std::vector<torch::Tensor> xs = x.split(DLSHOGI_FEATURES1_NUM, 1);
    torch::Tensor x1 = xs[0];
    torch::Tensor x2 = xs[1];

    auto out = module_.forward({ x1, x2 });
    auto tuple = out.toTuple();
    torch::Tensor policy_logits = tuple->elements()[0].toTensor();
    torch::Tensor value = tuple->elements()[1].toTensor();

    torch::Tensor policy_target = torch::tensor(policy_teachers).to(device_).view({ -1, POLICY_DIM });
    torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(policy_logits, 1), 1, false);

    torch::Tensor value_t = torch::tensor(value_teachers).to(device_);
    value = value.view(-1);
#ifdef USE_SIGMOID
    torch::Tensor value_loss = torch::binary_cross_entropy(value, value_t, {}, torch::Reduction::None);
#else
    value = value * 2 - 1;
    torch::Tensor value_loss = torch::mse_loss(value, value_t, torch::Reduction::None);
#endif

    return { policy_loss, value_loss };
#endif
}