#include "proposed_model_lstm.hpp"
#include "../../common.hpp"
#include "../common.hpp"

//ネットワークの設定
static constexpr int32_t HIDDEN_SIZE = 512;
static constexpr int32_t NUM_LAYERS = 1;

ProposedModelLSTMImpl::ProposedModelLSTMImpl(SearchOptions search_options) : BaseModel(search_options) {
    torch::nn::LSTMOptions option(StateEncoderImpl::HIDDEN_DIM, HIDDEN_SIZE);
    option.num_layers(NUM_LAYERS);
    readout_lstm_ = register_module("readout_lstm_", torch::nn::LSTM(option));
    readout_policy_head_ = register_module("readout_policy_head_", torch::nn::Linear(HIDDEN_SIZE, POLICY_DIM));
}

Move ProposedModelLSTMImpl::think(Position& root, int64_t time_limit) {
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

torch::Tensor ProposedModelLSTMImpl::embed(const std::vector<Position>& positions) {
    std::vector<float> features;
    for (const auto& position : positions) {
        std::vector<float> f = position.makeFeature();
        features.insert(features.end(), f.begin(), f.end());
    }
    torch::Tensor x = encoder_->embed(features, device_, fp16_, freeze_encoder_);
    x = x.view({ 1, (int64_t)positions.size(), StateEncoderImpl::HIDDEN_DIM });
    return x;
}

torch::Tensor ProposedModelLSTMImpl::simulationPolicy(const torch::Tensor& x) { return sim_policy_head_->forward(x); }

torch::Tensor ProposedModelLSTMImpl::readoutPolicy(const torch::Tensor& x) {
    //lstmは入力(input, (h_0, c_0))
    //inputのshapeは(seq_len, batch, input_size)
    //h_0, c_0は任意の引数で、状態を初期化できる
    //h_0, c_0のshapeは(num_layers_ * num_directions, batch, hidden_size_)

    //出力はoutput, (h_n, c_n)
    //outputのshapeは(seq_len, batch, num_directions * hidden_size)
    auto [output, h_and_c] = readout_lstm_->forward(x, std::make_tuple(readout_h_, readout_c_));
    std::tie(readout_h_, readout_c_) = h_and_c;

    torch::Tensor policy = readout_policy_head_->forward(output);

    return policy;
}

std::vector<torch::Tensor> ProposedModelLSTMImpl::loss(const std::vector<LearningData>& data) {
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
    torch::Tensor sim_policy_logit = simulationPolicy(x)[0];
    loss.push_back(policyLoss(sim_policy_logit, policy_teacher));

    //エントロピー正則化(Simulation Policyにかける)
    loss.push_back(entropyLoss(sim_policy_logit));

    return loss;
}

std::vector<torch::Tensor> ProposedModelLSTMImpl::search(std::vector<Position>& positions) {
    //探索をして出力方策の系列を得る
    std::vector<torch::Tensor> policy_logits;

    //バッチサイズを取得しておく
    const int64_t batch_size = positions.size();

    //状態を初期化
    //(num_layers * num_directions, batch, hidden_size)
    readout_h_ = torch::zeros({ NUM_LAYERS, batch_size, HIDDEN_SIZE }).to(device_);
    readout_c_ = torch::zeros({ NUM_LAYERS, batch_size, HIDDEN_SIZE }).to(device_);

    //何回探索したら戻すか
    constexpr int64_t RESET_NUM = 3;

    for (int64_t m = 0; m <= search_options_.search_limit; m++) {
        //盤面を戻す
        if (m != 0 && m % RESET_NUM == 0) {
            for (int64_t i = 0; i < batch_size; i++) {
                for (int64_t _ = 0; _ < RESET_NUM; _++) {
                    positions[i].undo();
                }
            }
        }

        //現局面の特徴を抽出
        torch::Tensor x = embed(positions);

        //ここまでの探索から最終行動決定
        policy_logits.push_back(readoutPolicy(x));

        if (m == search_options_.search_limit) {
            break;
        }

        //探索行動を決定
        torch::Tensor sim_policy_logit = simulationPolicy(x);

        //行動をサンプリングして盤面を動かす
        for (int64_t i = 0; i < batch_size; i++) {
            std::vector<Move> moves = positions[i].generateAllMoves();
            std::vector<float> logits(moves.size());
            for (uint64_t j = 0; j < moves.size(); j++) {
                logits[j] = sim_policy_logit[0][i][moves[j].toLabel()].item<float>();
            }
            std::vector<float> softmaxed = softmax(logits, 1.0f);
            int64_t move_index = randomChoose(softmaxed);
            positions[i].doMove(moves[move_index]);
        }
    }

    for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t _ = 0; _ < search_options_.search_limit % RESET_NUM; _++) {
            positions[i].undo();
        }
    }

    return policy_logits;
}