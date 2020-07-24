#include "stacked_lstm.hpp"
#include "../../common.hpp"

//ネットワークの設定
static constexpr int64_t BLOCK_NUM = 3;
static constexpr int32_t KERNEL_SIZE = 3;
static constexpr int32_t REDUCTION = 8;
static constexpr int32_t CHANNEL_NUM = 64;
static constexpr int64_t HIDDEN_CHANNEL_NUM = 32;
static constexpr int64_t HIDDEN_DIM = BOARD_WIDTH * BOARD_WIDTH * HIDDEN_CHANNEL_NUM;
static constexpr int32_t LSTM_HIDDEN_SIZE = 512;
static constexpr int32_t ABSTRACT_ACTION_DIM = 512;
static constexpr int32_t NUM_LAYERS = 1;

const std::string StackedLSTMImpl::MODEL_PREFIX = "stacked_lstm_bl" + std::to_string(BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
const std::string StackedLSTMImpl::DEFAULT_MODEL_NAME = StackedLSTMImpl::MODEL_PREFIX + ".model";

StackedLSTMImpl::StackedLSTMImpl(SearchOptions search_options) : search_options_(std::move(search_options)),
                                                                     blocks_(BLOCK_NUM, nullptr), device_(torch::kCUDA), fp16_(false) {
    first_conv_ = register_module("first_conv_", Conv2DwithBatchNorm(INPUT_CHANNEL_NUM, CHANNEL_NUM, KERNEL_SIZE));
    for (int32_t i = 0; i < BLOCK_NUM; i++) {
        blocks_[i] = register_module("blocks_" + std::to_string(i), ResidualBlock(CHANNEL_NUM, KERNEL_SIZE, REDUCTION));
    }
    last_conv_ = register_module("last_conv_", Conv2DwithBatchNorm(CHANNEL_NUM, HIDDEN_CHANNEL_NUM, KERNEL_SIZE));

    using torch::nn::LSTM;
    using torch::nn::LSTMOptions;
    env_model_lstm_ = register_module("env_model_lstm_", LSTM(LSTMOptions(ABSTRACT_ACTION_DIM, HIDDEN_DIM).num_layers(NUM_LAYERS)));

    LSTMOptions option(HIDDEN_DIM, LSTM_HIDDEN_SIZE);
    option.num_layers(NUM_LAYERS);
    simulation_lstm_ = register_module("simulation_lstm_", LSTM(option));
    simulation_policy_head_ = register_module("simulation_policy_head_", torch::nn::Linear(LSTM_HIDDEN_SIZE, ABSTRACT_ACTION_DIM));
    readout_lstm_ = register_module("readout_lstm_", LSTM(option));
    readout_policy_head_ = register_module("readout_policy_head_", torch::nn::Linear(LSTM_HIDDEN_SIZE, POLICY_DIM));
}

Move StackedLSTMImpl::think(Position& root, int64_t time_limit) {
    //思考を行う

    //状態を初期化
    //(num_layers * num_directions, batch, hidden_size)
    env_model_h_ = torch::zeros({ NUM_LAYERS, 1, HIDDEN_DIM }).to(device_);
    env_model_c_ = torch::zeros({ NUM_LAYERS, 1, HIDDEN_DIM }).to(device_);
    simulation_h_ = torch::zeros({ NUM_LAYERS, 1, LSTM_HIDDEN_SIZE }).to(device_);
    simulation_c_ = torch::zeros({ NUM_LAYERS, 1, LSTM_HIDDEN_SIZE }).to(device_);
    readout_h_ = torch::zeros({ NUM_LAYERS, 1, LSTM_HIDDEN_SIZE }).to(device_);
    readout_c_ = torch::zeros({ NUM_LAYERS, 1, LSTM_HIDDEN_SIZE }).to(device_);

    //出力系列を初期化
    outputs_.clear();

    //最初のエンコード
    torch::Tensor embed_vector = embed(root.makeFeature());

    //思考開始局面から考えた深さ
    int64_t depth = 0;

    for (int64_t m = 0; m < search_options_.search_limit; m++) {
        //今までの探索から現時点での結論を推論
        torch::Tensor readout_policy = readoutPolicy(embed_vector);
        outputs_.push_back(readout_policy);

        //LSTMでの探索を実行
        torch::Tensor abstract_action = simulationPolicy(embed_vector);

        //環境モデルに入力して次状態を予測
        auto[output, h_and_c] = env_model_lstm_->forward(abstract_action, std::make_tuple(env_model_h_, env_model_c_));
        std::tie(env_model_h_, env_model_c_) = h_and_c;

        //埋め込みベクトルを更新
        embed_vector = output;
    }

    //局面を戻す
    for (int64_t i = 0; i < depth; i++) {
        root.undo();
    }

    //合法手だけマスクをかける
    std::vector<Move> moves = root.generateAllMoves();
    std::vector<float> logits;
    for (const Move& move : moves) {
        logits.push_back(outputs_.back()[0][0][move.toLabel()].item<float>());
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

torch::Tensor StackedLSTMImpl::embed(const std::vector<float>& inputs) {
    torch::Tensor x = (fp16_ ? torch::tensor(inputs).to(device_, torch::kHalf) : torch::tensor(inputs).to(device_));
    x = x.view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
    torch::Tensor y = first_conv_->forward(x);
    for (ResidualBlock& block : blocks_) {
        y = block->forward(y);
    }
    y = last_conv_->forward(y);
    y = torch::flatten(y, 1);
    y = y.view({ 1, -1, HIDDEN_DIM });
    return y;
}

torch::Tensor StackedLSTMImpl::simulationPolicy(const torch::Tensor& x) {
    //lstmは入力(input, (h_0, c_0))
    //inputのshapeは(seq_len, batch, input_size)
    //h_0, c_0は任意の引数で、状態を初期化できる
    //h_0, c_0のshapeは(num_layers_ * num_directions, batch, hidden_size_)

    //出力はoutput, (h_n, c_n)
    //outputのshapeは(seq_len, batch, num_directions * hidden_size)
    auto[output, h_and_c] = simulation_lstm_->forward(x, std::make_tuple(simulation_h_, simulation_c_));
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
    auto[output, h_and_c] = readout_lstm_->forward(x, std::make_tuple(readout_h_, readout_c_));
    std::tie(readout_h_, readout_c_) = h_and_c;

    return readout_policy_head_->forward(output);
}

std::vector<torch::Tensor> StackedLSTMImpl::loss(const std::vector<LearningData>& data) {
    //現状バッチサイズは1のみに対応
    assert(data.size() == 1);

    Position root;
    root.fromStr(data.front().position_str);

    //探索を行い、各探索後の方策を得る(output_に保存される)
    think(root, INT_MAX);

    const int64_t M = outputs_.size();

    //policyの教師信号
    std::vector<float> policy_teachers(POLICY_DIM, 0.0);
    for (const std::pair<int32_t, float>& e : data.front().policy) {
        policy_teachers[e.first] = e.second;
    }

    torch::Tensor policy_teacher = torch::tensor(policy_teachers).to(device_).view({ -1, POLICY_DIM });

    //各探索後の損失を計算
    std::vector<torch::Tensor> l(M + 1);
    l[0] = torch::zeros({ 1 });
    for (int64_t m = 0; m < M; m++) {
        torch::Tensor policy_logit = outputs_[m][0];
        torch::Tensor log_softmax = torch::log_softmax(policy_logit, 1);
        torch::Tensor clipped = torch::clamp_min(log_softmax, -20);
        l[m + 1] = (-policy_teacher * clipped).sum();
    }

    //損失の差分を計算
    std::vector<torch::Tensor> r(M + 1);
    for (int64_t m = 0; m < M; m++) {
        r[m + 1] = -(l[m + 1] - l[m]);
    }

    //重み付き累積和
    constexpr float gamma = 1.0;
    std::vector<torch::Tensor> R(M + 1);
    for (int64_t m = 1; m <= M; m++) {
        R[m] = torch::zeros({ 1 });
        for (int64_t m2 = m; m2 <= M; m2++) {
            R[m] += std::pow(gamma, m2 - m) * r[m2];
        }

        //この値は勾配を切る
        R[m] = R[m].detach().to(device_);
    }

    std::vector<torch::Tensor> loss;
    loss.push_back(l[M].view({1}));
    for (int64_t m = 1; m <= M; m++) {
        loss.push_back(l[m].view({1}));
    }

    return loss;
}

void StackedLSTMImpl::setGPU(int16_t gpu_id, bool fp16) {
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    fp16_ = fp16;
    (fp16_ ? to(device_, torch::kHalf) : to(device_, torch::kFloat));
}