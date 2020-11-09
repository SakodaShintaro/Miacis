#include "loop_lstm.hpp"
#include "../../common.hpp"

//ネットワークの設定
static constexpr int32_t HIDDEN_DIM = StateEncoderImpl::HIDDEN_DIM;
static constexpr int32_t NUM_LAYERS = 2;

LoopLSTMImpl::LoopLSTMImpl(const SearchOptions& search_options) : BaseModel(search_options) {
    using namespace torch::nn;
    LSTMOptions option(HIDDEN_DIM, HIDDEN_DIM);
    option.num_layers(NUM_LAYERS);
    readout_lstm_ = register_module("readout_lstm_", LSTM(option));
    readout_policy_head_ = register_module("readout_policy_head_", Linear(HIDDEN_DIM, POLICY_DIM));
    readout_value_head_ = register_module("readout_value_head_", Linear(HIDDEN_DIM, 1));
}

std::vector<std::tuple<torch::Tensor, torch::Tensor>> LoopLSTMImpl::search(std::vector<Position>& positions) {
    //探索をして出力の系列を得る
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> policy_and_value;

    //最初のエンコード
    torch::Tensor embed_vector = embed(positions);

    //状態を初期化
    //(num_layers * num_directions, batch, hidden_size)
    const int64_t batch_size = positions.size();
    readout_h_ = torch::zeros({ NUM_LAYERS, batch_size, HIDDEN_DIM }).to(device_);
    readout_c_ = torch::zeros({ NUM_LAYERS, batch_size, HIDDEN_DIM }).to(device_);

    for (int64_t m = 0; m <= search_options_.search_limit; m++) {
        //lstmは入力(input, (h_0, c_0))
        //inputのshapeは(seq_len, batch, input_size)
        //h_0, c_0は任意の引数で、状態を初期化できる
        //h_0, c_0のshapeは(num_layers_ * num_directions, batch, hidden_size_)

        //出力はoutput, (h_n, c_n)
        //outputのshapeは(seq_len, batch, num_directions * hidden_size)
        auto [output, h_and_c] = readout_lstm_->forward(embed_vector, std::make_tuple(readout_h_, readout_c_));
        std::tie(readout_h_, readout_c_) = h_and_c;

        torch::Tensor policy_logit = readout_policy_head_->forward(output);
        torch::Tensor value = torch::tanh(readout_value_head_->forward(output));
        policy_and_value.emplace_back(policy_logit, value);

        embed_vector = output;
    }

    return policy_and_value;
}