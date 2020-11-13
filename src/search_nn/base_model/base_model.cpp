#include "base_model.hpp"
#include "../../common.hpp"
#include "../common.hpp"
#include <utility>

BaseModel::BaseModel(SearchOptions options)
    : search_options_(std::move(options)), device_(torch::kCUDA), fp16_(false), freeze_encoder_(true), last_only_(true) {
    encoder_ = register_module("encoder_", StateEncoder());
    base_policy_head_ = register_module("base_policy_head_", torch::nn::Linear(StateEncoderImpl::HIDDEN_DIM, POLICY_DIM));
    base_value_head_ = register_module("base_value_head_", torch::nn::Linear(StateEncoderImpl::HIDDEN_DIM, 1));
}

void BaseModel::setGPU(int16_t gpu_id, bool fp16) {
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    fp16_ = fp16;
    (fp16_ ? to(device_, torch::kHalf) : to(device_, torch::kFloat));
}

void BaseModel::loadPretrain(const std::string& encoder_path, const std::string& policy_head_path,
                             const std::string& value_head_path) {
    std::ifstream encoder_file(encoder_path);
    if (encoder_file.is_open()) {
        torch::load(encoder_, encoder_path);
    }
    std::ifstream policy_head_file(policy_head_path);
    if (policy_head_file.is_open()) {
        torch::load(base_policy_head_, policy_head_path);
    }
    std::ifstream value_head_file(value_head_path);
    if (value_head_file.is_open()) {
        torch::load(base_value_head_, value_head_path);
    }
}

void BaseModel::setOption(bool freeze_encoder, bool last_only) {
    freeze_encoder_ = freeze_encoder;
    last_only_ = last_only;
}

std::vector<torch::Tensor> BaseModel::loss(const std::vector<LearningData>& data) {
    //バッチサイズを取得しておく
    const int64_t batch_size = data.size();

    //盤面を復元
    std::vector<Position> positions(batch_size);
    for (int64_t i = 0; i < batch_size; i++) {
        positions[i].fromStr(data[i].position_str);
    }

    //探索をして出力方策の系列を得る
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> policy_and_value = search(positions);

    //教師信号の構築
    std::vector<float> policy_teachers(data.size() * POLICY_DIM, 0.0);
    std::vector<float> value_teachers;

    for (uint64_t i = 0; i < data.size(); i++) {
        //policyの教師信号
        for (const std::pair<int32_t, float>& e : data[i].policy) {
            policy_teachers[i * POLICY_DIM + e.first] = e.second;
        }

        //valueの教師信号
        value_teachers.push_back(data[i].value);
    }
    torch::Tensor policy_teacher = torch::tensor(policy_teachers).to(device_).view({ batch_size, POLICY_DIM });
    torch::Tensor value_teacher = torch::tensor(value_teachers).to(device_).view({ batch_size, 1 });

    //各探索後の損失を計算
    std::vector<torch::Tensor> loss;
    for (const auto& p_and_v : policy_and_value) {
        const auto& [policy_logit, value] = p_and_v;
        loss.push_back(policyLoss(policy_logit[0], policy_teacher));
        loss.push_back(torch::mse_loss(value, value_teacher));
    }

    //現局面に対するBaseの損失を計算する
    //現局面の特徴を抽出
    torch::Tensor x = embed(positions);

    //Policy損失
    torch::Tensor base_policy_logit = base_policy_head_->forward(x)[0];
    loss.push_back(policyLoss(base_policy_logit, policy_teacher));

    //Value損失
    torch::Tensor value = torch::tanh(base_value_head_->forward(x));
    loss.push_back(torch::mse_loss(value, value_teacher));

    //エントロピー正則化(Base Policyにかける)
    loss.push_back(entropyLoss(base_policy_logit));
    return loss;
}

torch::Tensor BaseModel::embed(const std::vector<Position>& positions) {
    std::vector<float> features;
    for (const auto& position : positions) {
        std::vector<float> f = position.makeFeature();
        features.insert(features.end(), f.begin(), f.end());
    }
    torch::Tensor x = encoder_->embed(features, device_, fp16_, freeze_encoder_);
    x = x.view({ 1, (int64_t)positions.size(), StateEncoderImpl::HIDDEN_DIM });
    return x;
}

Move BaseModel::think(Position& root, int64_t time_limit) {
    //思考を行う

    //投了判定
    float score{};
    if (root.isFinish(score) && score == MIN_SCORE) {
        return NULL_MOVE;
    }

    //バッチ化している関数と共通化している都合上、盤面をvector化
    std::vector<Position> positions;
    positions.push_back(root);

    //出力方策の系列を取得
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> policy_and_value = search(positions);

    //合法手だけマスクをかける
    std::vector<Move> moves = root.generateAllMoves();
    std::vector<float> logits;
    for (const Move& move : moves) {
        auto [policy_logit, value] = policy_and_value.back();
        logits.push_back(policy_logit[0][0][move.toLabel()].item<float>());
    }

    if (root.turnNumber() <= search_options_.random_turn) {
        //Softmaxの確率に従って選択
        std::vector<float> masked_policy = softmax(logits, 1.0f);
        int32_t move_id = randomChoose(masked_policy);
        return moves[move_id];
    } else {
        //最大のlogitを持つ行動を選択
        int32_t move_id = std::max_element(logits.begin(), logits.end()) - logits.begin();
        return moves[move_id];
    }
}