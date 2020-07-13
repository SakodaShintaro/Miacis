#include "proposed_lstm.hpp"
#include "../include_switch.hpp"

//ネットワークの設定
#ifdef SHOGI
static constexpr int32_t BLOCK_NUM = 10;
#elif OTHELLO
static constexpr int32_t BLOCK_NUM = 5;
#endif
static constexpr int32_t CHANNEL_NUM = 64;
static constexpr int32_t HIDDEN_SIZE = 512;
static constexpr int32_t NUM_LAYERS = 2;

ProposedModel::ProposedModel() {
    torch::nn::LSTMOptions option(BOARD_WIDTH * BOARD_WIDTH * CHANNEL_NUM, HIDDEN_SIZE);
    option.num_layers(NUM_LAYERS);
    lstm_ = register_module("lstm_", torch::nn::LSTM(option));
    policy_head_ = register_module("policy_head_", torch::nn::Linear(HIDDEN_SIZE, POLICY_DIM + 1));
    value_head_ = register_module("value_head_", torch::nn::Linear(HIDDEN_SIZE, 1));
    resetState();
}

std::tuple<torch::Tensor, torch::Tensor> ProposedModel::forward(const torch::Tensor& x) {
    //lstmは入力(input, (h_0, c_0))
    //inputのshapeは(seq_len, batch, input_size)
    //h_0, c_0は任意の引数で、状態を初期化できる
    //h_0, c_0のshapeは(num_layers_ * num_directions, batch, hidden_size_)

    //出力はoutput, (h_n, c_n)
    //outputのshapeは(seq_len, batch, num_directions * hidden_size)
    auto[output, h_and_c] = lstm_->forward(x, std::make_tuple(h_, c_));
    std::tie(h_, c_) = h_and_c;

    torch::Tensor policy = policy_head_->forward(output);
    torch::Tensor value = value_head_->forward(output);

    return std::make_tuple(policy, value);
}

torch::Tensor ProposedModel::loss(const torch::Tensor& x, const torch::Tensor& t) {
    auto[policy, value] = forward(x);

    //policyについての損失
    std::vector<float> losses;
    for (int64_t i = 0; i < x.size(0); i++) {
        //最終結果と比較して損失を貯めていく

    }

    //差分を計算する

    //差分が報酬のようなものになる
    //これを用いて方策勾配
    //今回選んだ行動と報酬の積をもとに勾配を計算

    return torch::Tensor();
}

void ProposedModel::resetState() {
    //(num_layers * num_directions, batch, hidden_size)
    h_ = torch::zeros({ NUM_LAYERS, 1, HIDDEN_SIZE });
    c_ = torch::zeros({ NUM_LAYERS, 1, HIDDEN_SIZE });
}