#include "simple_lstm.hpp"
#include "../../common.hpp"

//ネットワークの設定
static constexpr int32_t LSTM_HIDDEN_SIZE = 512;
static constexpr int32_t NUM_LAYERS = 1;

SimpleLSTMImpl::SimpleLSTMImpl(const SearchOptions& search_options) : BaseModel(search_options) {
    using namespace torch::nn;
    constexpr int64_t HIDDEN_DIM = StateEncoderImpl::HIDDEN_DIM;
    LSTMOptions option(HIDDEN_DIM, LSTM_HIDDEN_SIZE);
    option.num_layers(NUM_LAYERS);
    readout_lstm_ = register_module("readout_lstm_", LSTM(option));
    readout_policy_head_ = register_module("readout_policy_head_", Linear(LSTM_HIDDEN_SIZE, POLICY_DIM));
    readout_value_head_ = register_module("readout_value_head_", Linear(LSTM_HIDDEN_SIZE, 1));
}

std::tuple<torch::Tensor, torch::Tensor> SimpleLSTMImpl::readout(const torch::Tensor& x) {
    //lstmは入力(input, (h_0, c_0))
    //inputのshapeは(seq_len, batch, input_size)
    //h_0, c_0は任意の引数で、状態を初期化できる
    //h_0, c_0のshapeは(num_layers_ * num_directions, batch, hidden_size_)

    //出力はoutput, (h_n, c_n)
    //outputのshapeは(seq_len, batch, num_directions * hidden_size)
    auto [output, h_and_c] = readout_lstm_->forward(x, std::make_tuple(readout_h_, readout_c_));
    std::tie(readout_h_, readout_c_) = h_and_c;

    torch::Tensor policy_logit = readout_policy_head_->forward(output);
    torch::Tensor value = torch::tanh(readout_value_head_->forward(output));
    return std::make_tuple(policy_logit, value);
}

std::vector<std::tuple<torch::Tensor, torch::Tensor>> SimpleLSTMImpl::search(std::vector<Position>& positions) {
    //探索をして出力の系列を得る
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> policy_and_value;

    //最初のエンコード
    torch::Tensor embed_vector = embed(positions);

    //状態を初期化
    //(num_layers * num_directions, batch, hidden_size)
    const int64_t batch_size = positions.size();
    readout_h_ = torch::zeros({ NUM_LAYERS, batch_size, LSTM_HIDDEN_SIZE }).to(device_);
    readout_c_ = torch::zeros({ NUM_LAYERS, batch_size, LSTM_HIDDEN_SIZE }).to(device_);

    for (int64_t m = 0; m <= search_options_.search_limit; m++) {
        //常にルート局面の表現ベクトルを入力として推論
        if (!last_only_ || m == search_options_.search_limit) {
            policy_and_value.push_back(readout(embed_vector));
        } else {
            readout(embed_vector);
        }
    }

    return policy_and_value;
}