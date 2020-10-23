#include "base_model.hpp"
#include "../../common.hpp"
#include "../common.hpp"

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

std::vector<torch::Tensor> BaseModel::loss(const std::vector<LearningData>& data) {
    //バッチサイズを取得しておく
    const int64_t batch_size = data.size();

    //盤面を復元
    std::vector<Position> positions(batch_size);
    for (int64_t i = 0; i < batch_size; i++) {
        positions[i].fromStr(data[i].position_str);
    }

    //探索をして出力方策の系列を得る
    std::vector<torch::Tensor> policy_logits = search(positions);

    //policyの教師信号
    torch::Tensor policy_teacher = getPolicyTeacher(data, device_);

    //探索回数
    const int64_t M = search_options_.search_limit;

    //各探索後の損失を計算
    std::vector<torch::Tensor> loss(M + 1);
    for (int64_t m = 0; m <= M; m++) {
        torch::Tensor policy_logit = policy_logits[m][0]; //(batch_size, POLICY_DIM)
        loss[m] = policyLoss(policy_logit, policy_teacher);
    }

    //Simulation Policyの損失
    //現局面の特徴を抽出
    torch::Tensor x = embed(positions);
    torch::Tensor sim_policy_logit = sim_policy_head_->forward(x)[0];
    loss.push_back(policyLoss(sim_policy_logit, policy_teacher));

    //エントロピー正則化(Simulation Policyにかける)
    loss.push_back(entropyLoss(sim_policy_logit));
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
    std::vector<torch::Tensor> policy_logits = search(positions);

    //合法手だけマスクをかける
    std::vector<Move> moves = root.generateAllMoves();
    std::vector<float> logits;
    for (const Move& move : moves) {
        logits.push_back(policy_logits.back()[0][0][move.toLabel()].item<float>());
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