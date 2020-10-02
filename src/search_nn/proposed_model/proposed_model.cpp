#include "proposed_model.hpp"
#include "../../common.hpp"

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

std::vector<torch::Tensor> ProposedModelImpl::loss(const std::vector<LearningData>& data, bool freeze_encoder) {
    std::cerr << "No Implementation" << std::endl;
    std::exit(1);
}

void ProposedModelImpl::setGPU(int16_t gpu_id, bool fp16) {
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    fp16_ = fp16;
    (fp16_ ? to(device_, torch::kHalf) : to(device_, torch::kFloat));
}

void ProposedModelImpl::loadPretrain(const std::string& encoder_path, const std::string& policy_head_path) {
    torch::load(encoder_, encoder_path);
}