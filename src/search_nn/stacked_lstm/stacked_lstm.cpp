#include "stacked_lstm.hpp"
#include "../../common.hpp"
#include "../common.hpp"

//ネットワークの設定
static constexpr int64_t HIDDEN_DIM = BOARD_WIDTH * BOARD_WIDTH * StateEncoderImpl::LAST_CHANNEL_NUM;
static constexpr int32_t LSTM_HIDDEN_SIZE = 512;
static constexpr int32_t ABSTRACT_ACTION_DIM = 512;
static constexpr int32_t NUM_LAYERS = 1;

const std::string StackedLSTMImpl::MODEL_PREFIX = "stacked_lstm";
const std::string StackedLSTMImpl::DEFAULT_MODEL_NAME = StackedLSTMImpl::MODEL_PREFIX + ".model";

StackedLSTMImpl::StackedLSTMImpl(SearchOptions search_options)
    : search_options_(std::move(search_options)), device_(torch::kCUDA), fp16_(false) {
    encoder_ = register_module("encoder_", StateEncoder());

    using torch::nn::Linear;
    using torch::nn::LSTM;
    using torch::nn::LSTMOptions;

    env_model_lstm_ = register_module("env_model_lstm_", Linear(HIDDEN_DIM + ABSTRACT_ACTION_DIM, HIDDEN_DIM));

    LSTMOptions option(HIDDEN_DIM, LSTM_HIDDEN_SIZE);
    option.num_layers(NUM_LAYERS);
    simulation_lstm_ = register_module("simulation_lstm_", LSTM(option));
    simulation_policy_head_ = register_module("simulation_policy_head_", Linear(LSTM_HIDDEN_SIZE, ABSTRACT_ACTION_DIM));
    readout_lstm_ = register_module("readout_lstm_", LSTM(option));
    readout_policy_head_ = register_module("readout_policy_head_", Linear(LSTM_HIDDEN_SIZE, POLICY_DIM));
}

Move StackedLSTMImpl::think(Position& root, int64_t time_limit) {
    //思考を行う

    //状態を初期化
    //(num_layers * num_directions, batch, hidden_size)
    simulation_h_ = torch::zeros({ NUM_LAYERS, 1, LSTM_HIDDEN_SIZE }).to(device_);
    simulation_c_ = torch::zeros({ NUM_LAYERS, 1, LSTM_HIDDEN_SIZE }).to(device_);
    readout_h_ = torch::zeros({ NUM_LAYERS, 1, LSTM_HIDDEN_SIZE }).to(device_);
    readout_c_ = torch::zeros({ NUM_LAYERS, 1, LSTM_HIDDEN_SIZE }).to(device_);

    //出力系列を初期化
    outputs_.clear();

    //最初のエンコード
    torch::Tensor embed_vector = embed(root.makeFeature());

    //探索前の結果
    outputs_.push_back(readoutPolicy(embed_vector));

    for (int64_t m = 1; m <= search_options_.search_limit; m++) {
        //LSTMでの探索を実行
        torch::Tensor abstract_action = simulationPolicy(embed_vector);

        //環境モデルに入力して次状態を予測
        torch::Tensor c = torch::cat({ embed_vector, abstract_action }, 2);
        embed_vector = env_model_lstm_->forward(c);

        //今までの探索から現時点での結論を推論
        outputs_.push_back(readoutPolicy(embed_vector));
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
    torch::Tensor x = encoder_->embed(inputs, device_, fp16_, freeze_encoder_);
    x = x.view({ 1, -1, HIDDEN_DIM });
    return x;
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

std::vector<torch::Tensor> StackedLSTMImpl::loss(const std::vector<LearningData>& data, bool use_policy_gradient) {
    //現状バッチサイズは1のみに対応
    assert(data.size() == 1);

    Position root;
    root.fromStr(data.front().position_str);

    //探索を行い、各探索後の方策を得る(output_に保存される)
    think(root, INT_MAX);

    const int64_t M = search_options_.search_limit;

    //policyの教師信号
    torch::Tensor policy_teacher = getPolicyTeacher(data, device_);

    //各探索後の損失を計算
    std::vector<torch::Tensor> l(M + 1);
    for (int64_t m = 0; m <= M; m++) {
        torch::Tensor policy_logit = outputs_[m][0];
        torch::Tensor log_softmax = torch::log_softmax(policy_logit, 1);
        torch::Tensor clipped = torch::clamp_min(log_softmax, -20);
        l[m] = (-policy_teacher * clipped).sum();
    }

    if (!use_policy_gradient) {
        return l;
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
    for (int64_t m = 1; m <= M; m++) {
        loss.push_back(l[m]);
    }

    return loss;
}

std::vector<torch::Tensor> StackedLSTMImpl::lossBatch(const std::vector<LearningData>& data, bool use_policy_gradient) {
    //バッチサイズを取得しておく
    const int64_t batch_size = data.size();

    //探索回数
    const int64_t M = search_options_.search_limit;

    //盤面を復元
    std::vector<float> root_features;
    std::vector<Position> positions(batch_size);
    for (uint64_t i = 0; i < batch_size; i++) {
        positions[i].fromStr(data[i].position_str);
        std::vector<float> f = positions[i].makeFeature();
        root_features.insert(root_features.end(), f.begin(), f.end());
    }

    //GPUで計算
    torch::Tensor embed_vector = encoder_->embed(root_features, device_, fp16_, freeze_encoder_).cpu();

    //探索中に情報を落としておかなきゃいけないもの
    std::vector<std::vector<torch::Tensor>> outputs(batch_size), log_probs(batch_size);

    //思考を行う

    //状態を初期化
    //(num_layers * num_directions, batch, hidden_size)
    simulation_h_ = torch::zeros({ NUM_LAYERS, batch_size, LSTM_HIDDEN_SIZE }).to(device_);
    simulation_c_ = torch::zeros({ NUM_LAYERS, batch_size, LSTM_HIDDEN_SIZE }).to(device_);
    readout_h_ = torch::zeros({ NUM_LAYERS, batch_size, LSTM_HIDDEN_SIZE }).to(device_);
    readout_c_ = torch::zeros({ NUM_LAYERS, batch_size, LSTM_HIDDEN_SIZE }).to(device_);

    //探索前の結果
    outputs_.push_back(readoutPolicy(embed_vector));

    for (int64_t m = 1; m <= search_options_.search_limit; m++) {
        //LSTMでの探索を実行
        torch::Tensor abstract_action = simulationPolicy(embed_vector);

        //環境モデルに入力して次状態を予測
        torch::Tensor c = torch::cat({ embed_vector, abstract_action }, 1);
        embed_vector = env_model_lstm_->forward(c);

        //今までの探索から現時点での結論を推論
        outputs_.push_back(readoutPolicy(embed_vector));
    }

    std::cout << outputs_.back().sizes() << std::endl;
    std::exit(0);
}

void StackedLSTMImpl::setGPU(int16_t gpu_id, bool fp16) {
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    fp16_ = fp16;
    (fp16_ ? to(device_, torch::kHalf) : to(device_, torch::kFloat));
}

void StackedLSTMImpl::loadPretrain(const std::string& encoder_path, const std::string& policy_head_path) {
    torch::load(encoder_, encoder_path);
}

void StackedLSTMImpl::setOption(bool freeze_encoder, float gamma) { freeze_encoder_ = freeze_encoder; }