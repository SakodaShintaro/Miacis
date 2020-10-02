#include "stacked_lstm.hpp"
#include "../../common.hpp"

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

    action_encoder_ = register_module("action_encoder_", Linear(POLICY_DIM, ABSTRACT_ACTION_DIM));

    value_head_ = register_module("value_head_", Linear(HIDDEN_DIM, 1));
    policy_head_ = register_module("policy_head_", Linear(HIDDEN_DIM, POLICY_DIM));
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

    static Position pos;

    //局面の特徴量を取得
    std::array<std::vector<float>, LEARNING_RANGE> state_features;

    //Policyの教師
    std::array<std::vector<int64_t>, LEARNING_RANGE> move_teachers;

    //Valueの教師
    std::array<std::vector<ValueTeacherType>, LEARNING_RANGE> value_teachers;

    //行動の表現
    std::array<std::vector<Move>, LEARNING_RANGE> moves;

    for (const LearningData& datum : data) {
        //現局面の特徴量を取得
        pos.fromStr(datum.position_str);

        for (int64_t i = 0; i < LEARNING_RANGE; i++) {
            std::vector<float> curr_state_feature = pos.makeFeature();
            state_features[i].insert(state_features[i].end(), curr_state_feature.begin(), curr_state_feature.end());

            //Policyの教師
            move_teachers[i].push_back(datum.moves[i].toLabel());

            //Valueの教師
            value_teachers[i].push_back(datum.value[i]);

            //行動の表現取得のため
            moves[i].push_back(datum.moves[i]);

            //次の局面へ遷移
            pos.doMove(datum.moves[i]);
        }
    }

    //状態を初期化
    //(num_layers * num_directions, batch, hidden_size)
    simulation_h_ = torch::zeros({ NUM_LAYERS, batch_size, LSTM_HIDDEN_SIZE }).to(device_);
    simulation_c_ = torch::zeros({ NUM_LAYERS, batch_size, LSTM_HIDDEN_SIZE }).to(device_);
    readout_h_ = torch::zeros({ NUM_LAYERS, batch_size, LSTM_HIDDEN_SIZE }).to(device_);
    readout_c_ = torch::zeros({ NUM_LAYERS, batch_size, LSTM_HIDDEN_SIZE }).to(device_);

    //--------------------
    //    探索による損失
    //--------------------
    //探索をして出力方策の系列を得る
    std::vector<torch::Tensor> policy_logits = search(state_features[0]);

    //policyの教師信号
    torch::Tensor policy_teacher = torch::tensor(move_teachers[0]).to(device_);

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

        //nll_lossを使うかも
        //torch::nll_loss(torch::log_softmax(policy, 1), policy_teacher_tensor, {}, Reduction::None);

        //探索なしの直接推論についてエントロピーも求めておく
        if (m == 0) {
            entropy = (torch::softmax(policy_logit, 1) * clipped).sum(1).mean(0);
        }
    }

    //-----------------
    //    Value損失
    //-----------------
    //valueの教師信号を構築
    torch::Tensor value_teacher = torch::tensor(value_teachers[0]).to(device_);

    //価値を推定
    torch::Tensor embed_vector = embed(state_features[0]);
    torch::Tensor value = torch::tanh(value_head_->forward(embed_vector));
    value = value.view_as(value_teacher);

    //損失計算
    torch::Tensor value_loss = torch::mse_loss(value, value_teacher);
    loss.push_back(value_loss);

    //-----------------
    //    遷移予測部
    //-----------------

    //通常の損失
    for (int64_t i = 0; i < LEARNING_RANGE; i++) {
        torch::Tensor state_representation = embed(state_features[i]);
        loss.push_back(policyLoss(state_representation, move_teachers[i]));
        loss.push_back(valueLoss(state_representation, value_teachers[i]));
    }

    //遷移を介した損失
    //遷移予測した表現。最初は現局面の表現で初期化
    torch::Tensor simulated_representation = embed(state_features[0]);
    for (int64_t i = 1; i < LEARNING_RANGE; i++) {
        //行動の表現を取得
        torch::Tensor action_representation = encodeActions(moves[i]);

        //次状態を予測
        simulated_representation = predictNextState(simulated_representation, action_representation);

        loss.push_back(policyLoss(simulated_representation, move_teachers[i + 1]));
        loss.push_back(valueLoss(simulated_representation, value_teachers[i + 1]));
    }

    //-----------------------
    //    エントロピー正則化
    //-----------------------
    loss.push_back(entropy);

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

torch::Tensor StackedLSTMImpl::encodeActions(const std::vector<Move>& moves) {
    std::vector<float> move_features(moves.size() * POLICY_DIM, 0.0);
    for (uint64_t i = 0; i < moves.size(); i++) {
        move_features[i * POLICY_DIM + moves[i].toLabel()] = 1.0;
    }
    torch::Tensor move_features_tensor = torch::tensor(move_features).to(device_).view({ -1, POLICY_DIM });
    return action_encoder_->forward(move_features_tensor);
}

torch::Tensor StackedLSTMImpl::policyLoss(const torch::Tensor& state_representation, const std::vector<int64_t>& policy_teacher) {
    torch::Tensor policy_teacher_tensor = torch::tensor(policy_teacher).to(device_);
    torch::Tensor policy_logit = policy_head_->forward(state_representation).view_as(policy_teacher_tensor);
    return torch::nll_loss(torch::log_softmax(policy_logit, 1), policy_teacher_tensor);
}
torch::Tensor StackedLSTMImpl::valueLoss(const torch::Tensor& state_representation,
                                         const std::vector<ValueTeacherType>& value_teacher) {
    //Valueを取得
    torch::Tensor value = value_head_->forward(state_representation);
    value = torch::tanh(value);

    //教師の構築
    torch::Tensor value_teacher_tensor = torch::tensor(value_teacher).to(device_);

    //損失計算
#ifdef USE_CATEGORICAL
    return torch::nll_loss(torch::log_softmax(value, 1), value_teacher_tensor);
#else
#ifdef USE_SIGMOID
    return -value_teacher_tensor * torch::log(value) - (1 - value_teacher_tensor) * torch::log(1 - value);
#else
    return torch::mse_loss(value, value_teacher_tensor);
#endif
#endif
}