#include "infer_model.hpp"
#include "common.hpp"
#include "dataset.hpp"
#include "include_switch.hpp"
#include <torch/torch.h>
#include <trtorch/ptq.h>
#include <trtorch/trtorch.h>

void InferModel::load(const std::string& model_path, int64_t gpu_id, int64_t opt_batch_size,
                      const std::string& calibration_kifu_path, bool use_fp16) {
    //マルチGPU環境で同時にloadすると時々Segmentation Faultが発生するので排他制御を入れる
    static std::mutex load_mutex;
    std::lock_guard<std::mutex> guard(load_mutex);

    torch::jit::Module module = torch::jit::load(model_path);
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    module.to(device_);
    module.eval();

    std::vector<int64_t> in_min = { 1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH };
    std::vector<int64_t> in_opt = { opt_batch_size, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH };
    std::vector<int64_t> in_max = { opt_batch_size * 2, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH };

    use_fp16_ = use_fp16;
    if (use_fp16_) {
        trtorch::CompileSpec::InputRange range(in_min, in_opt, in_max);
        trtorch::CompileSpec info({ range });
        info.op_precision = torch::kHalf;
        info.device.gpu_id = gpu_id;
        module_ = trtorch::CompileGraph(module, info);
    } else {
        using namespace torch::data;
        auto dataset = CalibrationDataset(calibration_kifu_path, opt_batch_size * 2).map(transforms::Stack<>());
        auto dataloader = make_data_loader(std::move(dataset), DataLoaderOptions().batch_size(opt_batch_size).workers(1));

        const std::string name = "calibration_cache_file.txt";
        auto calibrator = trtorch::ptq::make_int8_calibrator<nvinfer1::IInt8MinMaxCalibrator>(std::move(dataloader), name, false);

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

std::pair<std::vector<PolicyType>, std::vector<ValueType>> InferModel::policyAndValueBatch(const std::vector<float>& inputs) {
    return tensorToVector(infer(inputs));
}

std::tuple<torch::Tensor, torch::Tensor> InferModel::infer(const std::vector<float>& inputs) {
    torch::Tensor x = torch::tensor(inputs).to(device_);
    x = x.view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
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

std::array<torch::Tensor, LOSS_TYPE_NUM> InferModel::validLoss(const std::vector<LearningData>& data) {
#ifdef USE_CATEGORICAL
    Position pos;
    std::vector<float> inputs;
    std::vector<float> policy_teachers(data.size() * POLICY_DIM, 0.0);
    std::vector<float> value_teachers;

    for (uint64_t i = 0; i < data.size(); i++) {
        pos.fromStr(data[i].position_str);

        //入力
        const std::vector<float> feature = pos.makeFeature();
        inputs.insert(inputs.end(), feature.begin(), feature.end());

        //policyの教師信号
        for (const std::pair<int32_t, float>& e : data[i].policy) {
            policy_teachers[i * POLICY_DIM + e.first] = e.second;
        }

        //valueの教師信号
        if (data[i].value != 0 && data[i].value != BIN_SIZE - 1) {
            std::cerr << "Categoricalの検証データは現状のところValueが-1 or 1でないといけない" << std::endl;
            std::exit(1);
        }
        value_teachers.push_back(data[i].value == 0 ? MIN_SCORE : MAX_SCORE);
    }

    torch::Tensor x = torch::tensor(inputs).to(device_);
    x = x.view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
    auto out = module_.forward({ x });
    auto tuple = out.toTuple();
    torch::Tensor policy_logit = tuple->elements()[0].toTensor();
    torch::Tensor value_logit = tuple->elements()[1].toTensor();

    torch::Tensor logits = policy_logit.view({ -1, POLICY_DIM });

    torch::Tensor policy_target = torch::tensor(policy_teachers).to(device_).view({ -1, POLICY_DIM });

    torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(logits, 1), 1, false);

    //Valueの分布を取得
    torch::Tensor value_cat = torch::softmax(value_logit, 1);

    //i番目の要素が示す値はMIN_SCORE + (i + 0.5) * VALUE_WIDTH
    std::vector<float> each_value;
    for (int64_t i = 0; i < BIN_SIZE; i++) {
        each_value.emplace_back(MIN_SCORE + (i + 0.5) * VALUE_WIDTH);
    }
    torch::Tensor each_value_tensor = torch::tensor(each_value).to(device_);

    //Categorical分布と内積を取ることで期待値を求める
    torch::Tensor value = (each_value_tensor * value_cat).sum(1);

    torch::Tensor value_t = torch::tensor(value_teachers).to(device_);

#ifdef USE_SIGMOID
    torch::Tensor value_loss = torch::binary_cross_entropy(value, value_t, {}, torch::Reduction::None);
#else
    torch::Tensor value_loss = torch::mse_loss(value, value_t, torch::Reduction::None);
#endif
    return { policy_loss, value_loss };

#else
    static Position pos;
    std::vector<float> inputs;
    std::vector<float> policy_teachers(data.size() * POLICY_DIM, 0.0);
    std::vector<ValueTeacherType> value_teachers;

    for (uint64_t i = 0; i < data.size(); i++) {
        pos.fromStr(data[i].position_str);

        //入力
        const std::vector<float> feature = pos.makeFeature();
        inputs.insert(inputs.end(), feature.begin(), feature.end());

        //policyの教師信号
        for (const std::pair<int32_t, float>& e : data[i].policy) {
            policy_teachers[i * POLICY_DIM + e.first] = e.second;
        }

        //valueの教師信号
        value_teachers.push_back(data[i].value);
    }

    torch::Tensor x = torch::tensor(inputs).to(device_);
    x = x.view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
    auto out = module_.forward({ x });
    auto tuple = out.toTuple();
    torch::Tensor policy = tuple->elements()[0].toTensor();
    torch::Tensor value = tuple->elements()[1].toTensor();

    torch::Tensor policy_logits = policy.view({ -1, POLICY_DIM });
    torch::Tensor policy_target = torch::tensor(policy_teachers).to(device_).view({ -1, POLICY_DIM });
    torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(policy_logits, 1), 1, false);

    torch::Tensor value_t = torch::tensor(value_teachers).to(device_);
    value = value.view(-1);
#ifdef USE_SIGMOID
    torch::Tensor value_loss = torch::binary_cross_entropy(value, value_t, {}, torch::Reduction::None);
#else
    torch::Tensor value_loss = torch::mse_loss(value, value_t, torch::Reduction::None);
#endif

    return { policy_loss, value_loss };
#endif
}