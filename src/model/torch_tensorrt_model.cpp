#include "torch_tensorrt_model.hpp"
#include "../common.hpp"
#include "../learn/learn.hpp"
#include "../shogi/position.hpp"
#include <torch/torch.h>
#include <torch_tensorrt/torch_tensorrt.h>

void TorchTensorRTModel::load(int64_t gpu_id, const SearchOptions& search_option) {
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
    torch_tensorrt::Input input(in_min, in_opt, in_max, torch::kFloat16);
    torch_tensorrt::ts::CompileSpec compile_spec({ input });
    compile_spec.enabled_precisions = { torch::kFloat16 };
    compile_spec.require_full_compilation = true;
    compile_spec.device.gpu_id = gpu_id;
    module_ = torch_tensorrt::ts::compile(module, compile_spec);
}

std::pair<std::vector<PolicyType>, std::vector<ValueType>>
tensorToVector(const std::tuple<torch::Tensor, torch::Tensor>& output) {
    const auto& [policy, value] = output;
    uint64_t batch_size = policy.size(0);

    std::vector<PolicyType> policies(batch_size);
    std::vector<ValueType> values(batch_size);

    if (policy.dtype() == torch::kFloat16) {
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
    //valueの方はfp16化してもなぜかHalfではなくFloatとして返ってくる
    //ひょっとしたらTRTorchのバグかも
    float* value_p = value.data_ptr<float>();
    for (uint64_t i = 0; i < batch_size; i++) {
        std::copy(value_p + i * BIN_SIZE, value_p + (i + 1) * BIN_SIZE, values[i].begin());
    }
#else
    std::copy(value.data_ptr<float>(), value.data_ptr<float>() + batch_size, values.begin());
#endif
    return std::make_pair(policies, values);
}

std::pair<std::vector<PolicyType>, std::vector<ValueType>>
TorchTensorRTModel::policyAndValueBatch(const std::vector<float>& inputs) {
    return tensorToVector(infer(inputs));
}

std::tuple<torch::Tensor, torch::Tensor> TorchTensorRTModel::infer(const std::vector<float>& inputs) {
    torch::Tensor x = inputVectorToTensor(inputs).to(device_);
    x = x.to(torch::kFloat16);
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

std::array<torch::Tensor, LOSS_TYPE_NUM> TorchTensorRTModel::validLoss(const std::vector<LearningData>& data) {
    auto [input, policy_target, value_target] = learningDataToTensor(data, device_);
    input = input.to(torch::kFloat16);
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
