#include "stacked_lstm.hpp"
#include "../../common.hpp"
#include "../common.hpp"

//ネットワークの設定
static constexpr int32_t LSTM_HIDDEN_SIZE = 512;
static constexpr int32_t ABSTRACT_ACTION_DIM = 512;
static constexpr int32_t NUM_LAYERS = 2;

StackedLSTMImpl::StackedLSTMImpl(const SearchOptions& search_options) : BaseModel(search_options) {
    using namespace torch::nn;

    constexpr int64_t HIDDEN_DIM = StateEncoderImpl::HIDDEN_DIM;
    env_model0_ = register_module("env_model0_", Linear(HIDDEN_DIM + ABSTRACT_ACTION_DIM, HIDDEN_DIM));
    env_model1_ = register_module("env_model1_", Linear(HIDDEN_DIM, HIDDEN_DIM));

    LSTMOptions option(HIDDEN_DIM, LSTM_HIDDEN_SIZE);
    option.num_layers(NUM_LAYERS);
    simulation_lstm_ = register_module("simulation_lstm_", LSTM(option));
    simulation_policy_head_ = register_module("simulation_policy_head_", Linear(LSTM_HIDDEN_SIZE, ABSTRACT_ACTION_DIM));
    readout_lstm_ = register_module("readout_lstm_", LSTM(option));
    readout_policy_head_ = register_module("readout_policy_head_", Linear(LSTM_HIDDEN_SIZE, POLICY_DIM));
}

torch::Tensor StackedLSTMImpl::simulationPolicy(const torch::Tensor& x) {
    //lstmは入力(input, (h_0, c_0))
    //inputのshapeは(seq_len, batch, input_size)
    //h_0, c_0は任意の引数で、状態を初期化できる
    //h_0, c_0のshapeは(num_layers_ * num_directions, batch, hidden_size_)

    //出力はoutput, (h_n, c_n)
    //outputのshapeは(seq_len, batch, num_directions * hidden_size)
    auto [output, h_and_c] = simulation_lstm_->forward(x, std::make_tuple(simulation_h_, simulation_c_));
    std::tie(simulation_h_, simulation_c_) = h_and_c;

    return simulation_policy_head_->forward(output);
}

torch::Tensor StackedLSTMImpl::readoutPolicy(const torch::Tensor& x) {
    //lstmは入力(input, (h_0, c_0))
    //inputのshapeは(seq_len, batch, input_size)
    //h_0, c_0は任意の引数で、状態を初期化できる
    //h_0, c_0のshapeは(num_layers_ * num_directions, batch, hidden_size_)

    //出力はoutput, (h_n, c_n)
    //outputのshapeは(seq_len, batch, num_directions * hidden_size)
    auto [output, h_and_c] = readout_lstm_->forward(x, std::make_tuple(readout_h_, readout_c_));
    std::tie(readout_h_, readout_c_) = h_and_c;

    return readout_policy_head_->forward(output);
}

torch::Tensor StackedLSTMImpl::predictNextState(const torch::Tensor& pre_state, const torch::Tensor& abstract_action) {
    torch::Tensor x = torch::cat({ pre_state, abstract_action }, 2);
    x = env_model0_->forward(x);
    x = torch::relu(x);
    x = env_model1_->forward(x);
    return pre_state + x;
}

std::vector<torch::Tensor> StackedLSTMImpl::search(std::vector<Position>& positions) {
    //探索をして出力方策の系列を得る
    std::vector<torch::Tensor> policy_logits;

    //バッチサイズを取得しておく
    const int64_t batch_size = positions.size();

    //最初のエンコード
    torch::Tensor embed_vector = embed(positions);

    //状態を初期化
    //(num_layers * num_directions, batch, hidden_size)
    simulation_h_ = torch::zeros({ NUM_LAYERS, batch_size, LSTM_HIDDEN_SIZE }).to(device_);
    simulation_c_ = torch::zeros({ NUM_LAYERS, batch_size, LSTM_HIDDEN_SIZE }).to(device_);
    readout_h_ = torch::zeros({ NUM_LAYERS, batch_size, LSTM_HIDDEN_SIZE }).to(device_);
    readout_c_ = torch::zeros({ NUM_LAYERS, batch_size, LSTM_HIDDEN_SIZE }).to(device_);

    //探索前の結果
    policy_logits.push_back(readoutPolicy(embed_vector));

    for (int64_t m = 1; m <= search_options_.search_limit; m++) {
        //Simulation Policyにより抽象的な行動を取得
        torch::Tensor abstract_action = simulationPolicy(embed_vector);

        //環境モデルに入力して次状態を予測
        embed_vector = predictNextState(embed_vector, abstract_action);

        //今までの探索から現時点での結論を推論
        policy_logits.push_back(readoutPolicy(embed_vector));
    }

    return policy_logits;
}