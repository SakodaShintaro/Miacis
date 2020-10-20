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

torch::Tensor ProposedModelLSTMImpl::readoutPolicy(const torch::Tensor& x, bool update_hidden_state) {
    //lstmは入力(input, (h_0, c_0))
    //inputのshapeは(seq_len, batch, input_size)
    //h_0, c_0は任意の引数で、状態を初期化できる
    //h_0, c_0のshapeは(num_layers_ * num_directions, batch, hidden_size_)

    //出力はoutput, (h_n, c_n)
    //outputのshapeは(seq_len, batch, num_directions * hidden_size)
    auto [output, h_and_c] = readout_lstm_->forward(x, std::make_tuple(readout_h_, readout_c_));

    if (update_hidden_state) {
        std::tie(readout_h_, readout_c_) = h_and_c;
    }

    return readout_policy_head_->forward(output);
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

    //ルート局面の特徴量
    torch::Tensor root_x = embed(positions);

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
        policy_logits.push_back(readoutPolicy(root_x, false));

        if (m == search_options_.search_limit) {
            break;
        }

        //現在の特徴を用いてLSTMの隠れ状態を更新
        readoutPolicy(x, true);

        //探索行動を決定
        torch::Tensor sim_policy_logit = sim_policy_head_->forward(x);

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