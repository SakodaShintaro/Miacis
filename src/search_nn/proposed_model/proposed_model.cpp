#include "proposed_model.hpp"
#include "../../common.hpp"
#include "../common.hpp"

//ネットワークの設定
static constexpr int64_t HIDDEN_DIM = BOARD_WIDTH * BOARD_WIDTH * StateEncoderImpl::LAST_CHANNEL_NUM;
static constexpr int32_t HIDDEN_SIZE = 512;
static constexpr int32_t NUM_LAYERS = 1;

const std::string ProposedModelImpl::MODEL_PREFIX = "proposed_model";
const std::string ProposedModelImpl::DEFAULT_MODEL_NAME = ProposedModelImpl::MODEL_PREFIX + ".model";

ProposedModelImpl::ProposedModelImpl(SearchOptions search_options)
    : search_options_(std::move(search_options)), device_(torch::kCUDA), fp16_(false) {
    encoder_ = register_module("encoder_", StateEncoder());

    torch::nn::LSTMOptions option(HIDDEN_DIM, HIDDEN_SIZE);
    option.num_layers(NUM_LAYERS);
    simulation_lstm_ = register_module("simulation_lstm_", torch::nn::LSTM(option));
    simulation_policy_head_ = register_module("simulation_policy_head_", torch::nn::Linear(HIDDEN_SIZE, POLICY_DIM + 1));
    simulation_value_head_ = register_module("simulation_value_head_", torch::nn::Linear(HIDDEN_SIZE, 1));
    readout_lstm_ = register_module("readout_lstm_", torch::nn::LSTM(option));
    readout_policy_head_ = register_module("readout_policy_head_", torch::nn::Linear(HIDDEN_SIZE, POLICY_DIM));
    readout_value_head_ = register_module("readout_value_head_", torch::nn::Linear(HIDDEN_SIZE, 1));
}

Move ProposedModelImpl::think(Position& root, int64_t time_limit) {
    //思考を行う

    //状態を初期化
    //(num_layers * num_directions, batch, hidden_size)
    simulation_h_ = torch::zeros({ NUM_LAYERS, 1, HIDDEN_SIZE }).to(device_);
    simulation_c_ = torch::zeros({ NUM_LAYERS, 1, HIDDEN_SIZE }).to(device_);
    readout_h_ = torch::zeros({ NUM_LAYERS, 1, HIDDEN_SIZE }).to(device_);
    readout_c_ = torch::zeros({ NUM_LAYERS, 1, HIDDEN_SIZE }).to(device_);

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
        torch::Tensor simulation_policy = simulationPolicy(embed_vector);

        //Policyからサンプリングして行動決定(undoを含む)
        std::vector<Move> moves = root.generateAllMoves();
        if (depth > 0) {
            //深さが1以上のときは1手戻るという選択肢を先頭に加える
            moves.insert(moves.begin(), NULL_MOVE);
        }
        std::vector<float> logits(moves.size());
        for (uint64_t i = 0; i < moves.size(); i++) {
            logits[i] = (depth > 0 && i == 0 ? simulation_policy[0][0][POLICY_DIM].item<float>()
                                             : simulation_policy[0][0][moves[i].toLabel()].item<float>());
        }
        std::vector<float> softmaxed = softmax(logits, 1.0f);
        int64_t move_index = randomChoose(softmaxed);

        //盤面を遷移
        if (depth > 0 && move_index == 0) {
            root.undo();
            depth--;
        } else {
            root.doMove(moves[move_index]);
            depth++;
        }

        //埋め込みベクトルを更新
        embed_vector = embed(root.makeFeature());
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

torch::Tensor ProposedModelImpl::embed(const std::vector<float>& inputs) {
    torch::Tensor x = encoder_->embed(inputs, device_, fp16_, freeze_encoder_);
    x = x.view({ 1, -1, HIDDEN_DIM });
    return x;
}

torch::Tensor ProposedModelImpl::simulationPolicy(const torch::Tensor& x) {
    //lstmは入力(input, (h_0, c_0))
    //inputのshapeは(seq_len, batch, input_size)
    //h_0, c_0は任意の引数で、状態を初期化できる
    //h_0, c_0のshapeは(num_layers_ * num_directions, batch, hidden_size_)

    //出力はoutput, (h_n, c_n)
    //outputのshapeは(seq_len, batch, num_directions * hidden_size)
    auto [output, h_and_c] = simulation_lstm_->forward(x, std::make_tuple(simulation_h_, simulation_c_));
    std::tie(simulation_h_, simulation_c_) = h_and_c;

    torch::Tensor policy = simulation_policy_head_->forward(output);

    return policy;
}

torch::Tensor ProposedModelImpl::readoutPolicy(const torch::Tensor& x) {
    //lstmは入力(input, (h_0, c_0))
    //inputのshapeは(seq_len, batch, input_size)
    //h_0, c_0は任意の引数で、状態を初期化できる
    //h_0, c_0のshapeは(num_layers_ * num_directions, batch, hidden_size_)

    //出力はoutput, (h_n, c_n)
    //outputのshapeは(seq_len, batch, num_directions * hidden_size)
    auto [output, h_and_c] = readout_lstm_->forward(x, std::make_tuple(readout_h_, readout_c_));
    std::tie(readout_h_, readout_c_) = h_and_c;

    torch::Tensor policy = readout_policy_head_->forward(output);

    return policy;
}

std::vector<torch::Tensor> ProposedModelImpl::loss(const std::vector<LearningData>& data, bool use_policy_gradient) {
    if (use_policy_gradient) {
        std::cout << "ProposedModel is not compatible with use_policy_gradient." << std::endl;
        std::exit(1);
    }

    //バッチサイズを取得しておく
    const int64_t batch_size = data.size();

    //盤面を復元
    std::vector<float> root_features;
    std::vector<Position> positions(batch_size);
    for (int64_t i = 0; i < batch_size; i++) {
        positions[i].fromStr(data[i].position_str);
        std::vector<float> f = positions[i].makeFeature();
        root_features.insert(root_features.end(), f.begin(), f.end());
    }

    //状態を初期化
    //(num_layers * num_directions, batch, hidden_size)
    simulation_h_ = torch::zeros({ NUM_LAYERS, batch_size, HIDDEN_SIZE }).to(device_);
    simulation_c_ = torch::zeros({ NUM_LAYERS, batch_size, HIDDEN_SIZE }).to(device_);
    readout_h_ = torch::zeros({ NUM_LAYERS, batch_size, HIDDEN_SIZE }).to(device_);
    readout_c_ = torch::zeros({ NUM_LAYERS, batch_size, HIDDEN_SIZE }).to(device_);

    //探索をして出力方策の系列を得る
    std::vector<torch::Tensor> policy_logits;

    //探索行動系列の履歴
    std::vector<std::set<std::vector<int32_t>>> moves_histories;

    //何回探索したら戻すか
    constexpr int64_t RESET_NUM = 3;

    for (int64_t m = 0; m <= search_options_.search_limit; m++) {
        if (m % RESET_NUM == 0) {
            //盤面を戻す
            for (int64_t i = 0; i < batch_size; i++) {
                for (int64_t _ = 0; _ < RESET_NUM; _++) {
                    positions[i].undo();
                }
            }
        }

        //現局面の特徴を抽出
        std::vector<float> features;
        for (int64_t i = 0; i < batch_size; i++) {
            std::vector<float> f = positions[i].makeFeature();
            features.insert(features.end(), f.begin(), f.end());
        }
        torch::Tensor x = embed(features);

        //ここまでの探索から最終行動決定
        policy_logits.push_back(readoutPolicy(x));

        //探索行動を決定
        torch::Tensor sim_policy_logit = (search_options_.use_readout_only ? readoutPolicy(x) : simulationPolicy(x));

        //行動をサンプリングして盤面を動かす
        for (int64_t i = 0; i < batch_size; i++) {
            std::vector<Move> moves = positions[i].generateAllMoves();
            std::vector<float> logits(moves.size());
            for (uint64_t j = 0; j < moves.size(); j++) {
                logits[j] = sim_policy_logit[0][j][moves[i].toLabel()].item<float>();
            }
            std::vector<float> softmaxed = softmax(logits, 1.0f);
            int64_t move_index = randomChoose(softmaxed);
            positions[i].doMove(moves[move_index]);
        }
    }

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
    return loss;
}

void ProposedModelImpl::setGPU(int16_t gpu_id, bool fp16) {
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    fp16_ = fp16;
    (fp16_ ? to(device_, torch::kHalf) : to(device_, torch::kFloat));
}

void ProposedModelImpl::loadPretrain(const std::string& encoder_path, const std::string& policy_head_path) {
    std::ifstream encoder_file(encoder_path);
    if (encoder_file.is_open()) {
        torch::load(encoder_, encoder_path);
    }
}

void ProposedModelImpl::setOption(bool freeze_encoder, float gamma) { freeze_encoder_ = freeze_encoder; }