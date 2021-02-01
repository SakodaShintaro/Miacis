#include "learning_model.hpp"
#include "common.hpp"
#include "include_switch.hpp"
#include <torch/torch.h>

void LearningModel::load(const std::string& model_path, int64_t gpu_id) {
    module_ = torch::jit::load(model_path);
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    module_.to(torch::kCUDA);
}

void LearningModel::save(const std::string& model_path) {
    module_.save(model_path);
}

torch::Tensor LearningModel::encode(const std::vector<float>& inputs) const {
    torch::Tensor x = torch::tensor(inputs).to(device_);
    x = x.view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
    return x;
}

std::array<torch::Tensor, LOSS_TYPE_NUM> LearningModel::loss(const std::vector<LearningData>& data) {
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

    torch::Tensor input_tensor = encode(inputs);
    auto out = module_.forward({ input_tensor });
    auto tuple = out.toTuple();
    torch::Tensor policy = tuple->elements()[0].toTensor();
    torch::Tensor value = tuple->elements()[1].toTensor();

    torch::Tensor policy_logits = policy.view({ -1, POLICY_DIM });
    torch::Tensor policy_target = torch::tensor(policy_teachers).to(device_).view({ -1, POLICY_DIM });
    torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(policy_logits, 1), 1, false);

#ifdef USE_CATEGORICAL
    torch::Tensor categorical_target = torch::tensor(value_teachers).to(device_);
    torch::Tensor value_loss = torch::nll_loss(torch::log_softmax(y.second, 1), categorical_target);
#else
    torch::Tensor value_t = torch::tensor(value_teachers).to(device_);
    value = value.view(-1);
#ifdef USE_SIGMOID
    torch::Tensor value_loss = torch::binary_cross_entropy(value, value_t, {}, torch::Reduction::None);
#else
    torch::Tensor value_loss = torch::mse_loss(value, value_t, torch::Reduction::None);
#endif
#endif

    return { policy_loss, value_loss };
}

std::array<torch::Tensor, LOSS_TYPE_NUM> LearningModel::validLoss(const std::vector<LearningData>& data) {
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

    std::pair<torch::Tensor, torch::Tensor> y = forward(inputs);
    torch::Tensor logits = y.first.view({ -1, POLICY_DIM });

    torch::Tensor policy_target =
        (fp16_ ? torch::tensor(policy_teachers).to(device_, torch::kHalf) : torch::tensor(policy_teachers).to(device_))
            .view({ -1, POLICY_DIM });

    torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(logits, 1), 1, false);

    //Valueの分布を取得
    torch::Tensor value_cat = torch::softmax(y.second, 1);

    //i番目の要素が示す値はMIN_SCORE + (i + 0.5) * VALUE_WIDTH
    std::vector<float> each_value;
    for (int64_t i = 0; i < BIN_SIZE; i++) {
        each_value.emplace_back(MIN_SCORE + (i + 0.5) * VALUE_WIDTH);
    }
    torch::Tensor each_value_tensor = torch::tensor(each_value).to(device_);

    //Categorical分布と内積を取ることで期待値を求める
    torch::Tensor value = (each_value_tensor * value_cat).sum(1);

    torch::Tensor value_t =
        (fp16_ ? torch::tensor(value_teachers).to(device_, torch::kHalf) : torch::tensor(value_teachers).to(device_));

#ifdef USE_SIGMOID
    torch::Tensor value_loss = torch::binary_cross_entropy(value, value_t, {}, torch::Reduction::None);
#else
    torch::Tensor value_loss = torch::mse_loss(value, value_t, torch::Reduction::None);
#endif

    return { policy_loss, value_loss };
#else
    //Scalarモデルの場合はloss関数と同じ
    return loss(data);
#endif
}

std::vector<torch::Tensor> LearningModel::parameters() {
    std::vector<torch::Tensor> parameters;
    for (auto p : module_.parameters()) {
        parameters.push_back(p);
    }
    return parameters;
}
