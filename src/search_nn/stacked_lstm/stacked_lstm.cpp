#include "stacked_lstm.hpp"
#include "../../common.hpp"
#include "../common.hpp"

//ネットワークの設定
static constexpr int64_t HIDDEN_DIM = BOARD_WIDTH * BOARD_WIDTH * StateEncoderImpl::LAST_CHANNEL_NUM;
static constexpr int32_t LSTM_HIDDEN_SIZE = 512;
static constexpr int32_t ABSTRACT_ACTION_DIM = 512;
static constexpr int32_t NUM_LAYERS = 2;

const std::string StackedLSTMImpl::MODEL_PREFIX = "stacked_lstm";
const std::string StackedLSTMImpl::DEFAULT_MODEL_NAME = StackedLSTMImpl::MODEL_PREFIX + ".model";

StackedLSTMImpl::StackedLSTMImpl(SearchOptions search_options)
    : search_options_(std::move(search_options)), device_(torch::kCUDA), fp16_(false), freeze_encoder_(true) {
    encoder_ = register_module("encoder_", StateEncoder());

    using torch::nn::Linear;
    using torch::nn::LSTM;
    using torch::nn::LSTMOptions;

    value_head_ = register_module("value_head_", Linear(HIDDEN_DIM, 1));
    env_model0_ = register_module("env_model0_", Linear(HIDDEN_DIM + ABSTRACT_ACTION_DIM, HIDDEN_DIM));
    env_model1_ = register_module("env_model1_", Linear(HIDDEN_DIM, HIDDEN_DIM));

    LSTMOptions option(HIDDEN_DIM + 1, LSTM_HIDDEN_SIZE);
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

    //探索をして出力方策の系列を得る
    std::vector<torch::Tensor> policy_logits = search(root.makeFeature());

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

std::vector<torch::Tensor> StackedLSTMImpl::search(const std::vector<float>& inputs) {
    //出力方策の系列
    std::vector<torch::Tensor> policy_logits;

    //最初のエンコード
    torch::Tensor embed_vector = embed(inputs);

    //価値を推定
    torch::Tensor value = torch::tanh(value_head_->forward(embed_vector));

    //表現と価値を連結
    torch::Tensor concatenated = torch::cat({ embed_vector, value }, -1);

    //探索前の結果
    policy_logits.push_back(readoutPolicy(concatenated));

    for (int64_t m = 1; m <= search_options_.search_limit; m++) {
        if (!search_options_.use_readout_only) {
            //Simulation Policyにより抽象的な行動を取得
            torch::Tensor abstract_action = simulationPolicy(concatenated);

            //環境モデルに入力して次状態を予測
            embed_vector = predictNextState(embed_vector, abstract_action);

            //価値を推定
            value = torch::tanh(value_head_->forward(embed_vector));

            //表現と価値を連結
            concatenated = torch::cat({ embed_vector, value }, -1);
        }

        //今までの探索から現時点での結論を推論
        policy_logits.push_back(readoutPolicy(concatenated));
    }

    return policy_logits;
}

torch::Tensor StackedLSTMImpl::predictNextState(const torch::Tensor& pre_state, const torch::Tensor& abstract_action) {
    torch::Tensor x = torch::cat({ pre_state, abstract_action }, 2);
    x = env_model0_->forward(x);
    x = torch::relu(x);
    x = env_model1_->forward(x);
    return pre_state + x;
}

std::vector<torch::Tensor> StackedLSTMImpl::loss(const std::vector<LearningData>& data, bool use_policy_gradient) {
    if (use_policy_gradient) {
        std::cout << "StackedLSTM is not compatible with use_policy_gradient." << std::endl;
        std::exit(1);
    }

    //バッチサイズを取得しておく
    const int64_t batch_size = data.size();

    //盤面を復元
    std::vector<float> root_features;
    Position pos;
    for (int64_t i = 0; i < batch_size; i++) {
        pos.fromStr(data[i].position_str);
        std::vector<float> f = pos.makeFeature();
        root_features.insert(root_features.end(), f.begin(), f.end());
    }

    //状態を初期化
    //(num_layers * num_directions, batch, hidden_size)
    simulation_h_ = torch::zeros({ NUM_LAYERS, batch_size, LSTM_HIDDEN_SIZE }).to(device_);
    simulation_c_ = torch::zeros({ NUM_LAYERS, batch_size, LSTM_HIDDEN_SIZE }).to(device_);
    readout_h_ = torch::zeros({ NUM_LAYERS, batch_size, LSTM_HIDDEN_SIZE }).to(device_);
    readout_c_ = torch::zeros({ NUM_LAYERS, batch_size, LSTM_HIDDEN_SIZE }).to(device_);

    //探索をして出力方策の系列を得る
    std::vector<torch::Tensor> policy_logits = search(root_features);

    //policyの教師信号
    torch::Tensor policy_teacher = getPolicyTeacher(data, device_);

    //エントロピー正則化の項
    torch::Tensor entropy;

    //探索回数
    const int64_t M = search_options_.search_limit;

    //各探索後の損失を計算
    std::vector<torch::Tensor> loss(M + 1);
    for (int64_t m = 0; m <= M; m++) {
        torch::Tensor policy_logit = policy_logits[m][0]; //(batch_size, POLICY_DIM)
        torch::Tensor log_softmax = torch::log_softmax(policy_logit, 1);
        torch::Tensor clipped = torch::clamp_min(log_softmax, -20);
        loss[m] = (-policy_teacher * clipped).sum(1).mean();

        //探索なしの直接推論についてエントロピーも求めておく
        if (m == 0) {
            entropy = (torch::softmax(policy_logit, 1) * clipped).sum(1).mean(0);
        }
    }

    loss.push_back(entropy);

    //価値を追加

    //valueの教師信号を構築
    std::vector<ValueTeacherType> value_teachers;
    for (const LearningData& datum : data) {
        value_teachers.push_back(datum.value);
    }
    torch::Tensor value_t =
        (fp16_ ? torch::tensor(value_teachers).to(device_, torch::kHalf) : torch::tensor(value_teachers).to(device_));

    //最初のエンコード
    torch::Tensor embed_vector = embed(root_features);

    //価値を推定
    torch::Tensor value = torch::tanh(value_head_->forward(embed_vector));

    torch::Tensor value_loss = torch::mse_loss(value, value_t);

    loss.push_back(value_loss);

    return loss;
}

void StackedLSTMImpl::setGPU(int16_t gpu_id, bool fp16) {
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    fp16_ = fp16;
    (fp16_ ? to(device_, torch::kHalf) : to(device_, torch::kFloat));
}

void StackedLSTMImpl::loadPretrain(const std::string& encoder_path, const std::string& policy_head_path,
                                   const std::string& value_head_path) {
    std::ifstream encoder_file(encoder_path);
    if (encoder_file.is_open()) {
        torch::load(encoder_, encoder_path);
    }
    //policy_headは特に読み込まない

    std::ifstream value_file(value_head_path);
    if (value_file.is_open()) {
        torch::load(value_head_, value_head_path);
    }
}

void StackedLSTMImpl::setOption(bool freeze_encoder, float gamma) { freeze_encoder_ = freeze_encoder; }