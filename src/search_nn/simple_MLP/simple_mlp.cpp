#include "simple_mlp.hpp"
#include "../../common.hpp"

//ネットワークの設定
static constexpr int64_t HIDDEN_DIM = BOARD_WIDTH * BOARD_WIDTH * StateEncoderImpl::LAST_CHANNEL_NUM;

const std::string SimpleMLPImpl::MODEL_PREFIX = "simple_mlp";
const std::string SimpleMLPImpl::DEFAULT_MODEL_NAME = SimpleMLPImpl::MODEL_PREFIX + ".model";

SimpleMLPImpl::SimpleMLPImpl(SearchOptions search_options)
    : search_options_(std::move(search_options)), device_(torch::kCUDA), fp16_(false) {
    encoder_ = register_module("encoder_", StateEncoder());
    policy_ = register_module("policy_", torch::nn::Linear(HIDDEN_DIM, POLICY_DIM));
}

Move SimpleMLPImpl::think(Position& root, int64_t time_limit) {
    //この局面を推論して探索せずにそのまま出力

    //ニューラルネットワークによる推論
    torch::Tensor policy = inferPolicy(root);

    //合法手だけマスクをかける
    std::vector<Move> moves = root.generateAllMoves();
    std::vector<float> logits;
    for (const Move& move : moves) {
        logits.push_back(policy[0][move.toLabel()].item<float>());
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

std::vector<torch::Tensor> SimpleMLPImpl::loss(const std::vector<LearningData>& data) {
    //現状バッチサイズは1のみに対応
    assert(data.size() == 1);

    //局面を構築して推論
    Position root;
    root.fromStr(data.front().position_str);
    torch::Tensor policy_logit = inferPolicy(root);

    //policyの教師信号
    std::vector<float> policy_teachers(POLICY_DIM, 0.0);
    for (const std::pair<int32_t, float>& e : data.front().policy) {
        policy_teachers[e.first] = e.second;
    }

    torch::Tensor policy_teacher = torch::tensor(policy_teachers).to(device_).view({ -1, POLICY_DIM });

    //損失を計算
    std::vector<torch::Tensor> loss(1);
    torch::Tensor log_softmax = torch::log_softmax(policy_logit, 1);
    torch::Tensor clipped = torch::clamp_min(log_softmax, -20);
    loss[0] = (-policy_teacher * clipped).sum().view({ 1 });

    return loss;
}

torch::Tensor SimpleMLPImpl::inferPolicy(const Position& pos) {
    std::vector<FloatType> inputs = pos.makeFeature();
    torch::Tensor x = (fp16_ ? torch::tensor(inputs).to(device_, torch::kHalf) : torch::tensor(inputs).to(device_));
    x = x.view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
    x = encoder_->forward(x);
    x = x.view({ -1, HIDDEN_DIM });
    return policy_->forward(x);
}

void SimpleMLPImpl::setGPU(int16_t gpu_id, bool fp16) {
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    fp16_ = fp16;
    (fp16_ ? to(device_, torch::kHalf) : to(device_, torch::kFloat));
}