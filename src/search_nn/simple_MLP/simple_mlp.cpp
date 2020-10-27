#include "simple_mlp.hpp"
#include "../../common.hpp"
#include "../common.hpp"

SimpleMLPImpl::SimpleMLPImpl(const SearchOptions& search_options) : BaseModel(search_options) {}

Move SimpleMLPImpl::think(Position& root, int64_t time_limit) {
    //この局面を推論して探索せずにそのまま出力

    //ニューラルネットワークによる推論
    auto [policy_logit, value] = infer(root);

    //合法手だけマスクをかける
    std::vector<Move> moves = root.generateAllMoves();
    std::vector<float> logits;
    for (const Move& move : moves) {
        logits.push_back(policy_logit[0][move.toLabel()].item<float>());
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

std::vector<torch::Tensor> SimpleMLPImpl::lossFunc(const std::vector<LearningData>& data) {
    const uint64_t batch_size = data.size();
    std::vector<float> inputs;
    std::vector<float> policy_teachers(POLICY_DIM * batch_size, 0.0);
    std::vector<float> value_teachers;
    static Position pos;
    for (uint64_t i = 0; i < batch_size; i++) {
        //入力
        pos.fromStr(data[i].position_str);
        std::vector<float> curr_feature = pos.makeFeature();
        inputs.insert(inputs.end(), curr_feature.begin(), curr_feature.end());

        //policyの教師信号
        for (const std::pair<int32_t, float>& e : data[i].policy) {
            policy_teachers[i * POLICY_DIM + e.first] = e.second;
        }

        //valueの教師信号
        value_teachers.push_back(data[i].value);
    }

    //推論
    torch::Tensor x = (fp16_ ? torch::tensor(inputs).to(device_, torch::kHalf) : torch::tensor(inputs).to(device_));
    auto [policy_logit, value] = forward(x);

    //損失
    std::vector<torch::Tensor> loss;

    //Policy損失:教師との交差エントロピー
    torch::Tensor policy_teacher = torch::tensor(policy_teachers).to(device_).view({ -1, POLICY_DIM });
    loss.push_back(policyLoss(policy_logit, policy_teacher));

    //Value損失:教師との自乗誤差
    torch::Tensor value_teacher = torch::tensor(value_teachers).to(device_);
    value = value.view_as(value_teacher);
    torch::Tensor value_loss = torch::mse_loss(value, value_teacher);
    loss.push_back(value_loss);

    //エントロピー正則化
    loss.push_back(entropyLoss(policy_logit));

    return loss;
}

std::tuple<torch::Tensor, torch::Tensor> SimpleMLPImpl::infer(const Position& pos) {
    std::vector<float> inputs = pos.makeFeature();
    torch::Tensor x = (fp16_ ? torch::tensor(inputs).to(device_, torch::kHalf) : torch::tensor(inputs).to(device_));
    return forward(x);
}

std::tuple<torch::Tensor, torch::Tensor> SimpleMLPImpl::forward(const torch::Tensor& x) {
    torch::Tensor y = x.view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
    y = encoder_->forward(y);
    y = y.view({ -1, StateEncoderImpl::HIDDEN_DIM });
    return std::make_tuple(base_policy_head_->forward(y), base_value_head_->forward(y));
}

void SimpleMLPImpl::save() {
    torch::save(encoder_, "encoder.model");
    torch::save(base_policy_head_, "policy_head.model");
    torch::save(base_value_head_, "value_head.model");
}

std::vector<torch::Tensor> SimpleMLPImpl::search(std::vector<Position>& positions) {
    std::cerr << "SimpleMLPではsearch関数は使わない" << std::endl;
    std::exit(1);
}