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

ProposedModelImpl::ProposedModelImpl() {
    torch::nn::LSTMOptions option(BOARD_WIDTH * BOARD_WIDTH * CHANNEL_NUM, HIDDEN_SIZE);
    option.num_layers(NUM_LAYERS);
    lstm_ = register_module("lstm_", torch::nn::LSTM(option));
    policy_head_ = register_module("policy_head_", torch::nn::Linear(HIDDEN_SIZE, POLICY_DIM + 1));
    value_head_ = register_module("value_head_", torch::nn::Linear(HIDDEN_SIZE, 1));
    resetState();
}

ProposedModelImpl::ProposedModelImpl(const SearchOptions& search_options) : search_options_(search_options) {
    torch::nn::LSTMOptions option(BOARD_WIDTH * BOARD_WIDTH * CHANNEL_NUM, HIDDEN_SIZE);
    option.num_layers(NUM_LAYERS);
    lstm_ = register_module("lstm_", torch::nn::LSTM(option));
    policy_head_ = register_module("policy_head_", torch::nn::Linear(HIDDEN_SIZE, POLICY_DIM + 1));
    value_head_ = register_module("value_head_", torch::nn::Linear(HIDDEN_SIZE, 1));
    resetState();
}

Move ProposedModelImpl::think(Position& root, int64_t time_limit, bool save_info_to_learn) {
    //思考を行う
    //時間制限、あるいはノード数制限に基づいて何回やるかを決める
    std::vector<torch::Tensor> outputs;

    //最初のエンコード
    torch::Tensor embed_vector;

    for (int64_t m = 0; m < search_options_.search_limit; m++) {
        //今までの探索から現時点での結論を推論


        //LSTMでの探索を実行
        auto[policy, value] = lstm_->forward(embed_vector, std::make_tuple(h_, c_));

        //Policyからサンプリングして行動決定(undoを含む)

        //盤面を遷移

        //埋め込みベクトルを更新
        embed_vector = embed_vector;
    }

    //合法手だけマスクをかける
    std::vector<Move> moves = root.generateAllMoves();
    std::vector<float> logits;
    for (const Move& move : moves) {
        logits.push_back(outputs.back()[0][move.toLabel()].item<float>());
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

std::tuple<torch::Tensor, torch::Tensor> ProposedModelImpl::forward(const torch::Tensor& x) {
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

torch::Tensor ProposedModelImpl::loss(const torch::Tensor& x, const torch::Tensor& t) {
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

void ProposedModelImpl::resetState() {
    //(num_layers * num_directions, batch, hidden_size)
    h_ = torch::zeros({ NUM_LAYERS, 1, HIDDEN_SIZE });
    c_ = torch::zeros({ NUM_LAYERS, 1, HIDDEN_SIZE });
}