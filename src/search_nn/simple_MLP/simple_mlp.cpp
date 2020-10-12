#include "simple_mlp.hpp"
#include "../../common.hpp"
#include "../common.hpp"

//ネットワークの設定
static constexpr int64_t HIDDEN_DIM = BOARD_WIDTH * BOARD_WIDTH * StateEncoderImpl::LAST_CHANNEL_NUM;

const std::string SimpleMLPImpl::MODEL_PREFIX = "simple_mlp";
const std::string SimpleMLPImpl::DEFAULT_MODEL_NAME = SimpleMLPImpl::MODEL_PREFIX + ".model";

SimpleMLPImpl::SimpleMLPImpl(SearchOptions search_options)
    : search_options_(std::move(search_options)), device_(torch::kCUDA), fp16_(false) {
    encoder = register_module("encoder", StateEncoder());
    policy_head = register_module("policy_head", torch::nn::Linear(HIDDEN_DIM, POLICY_DIM));
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
    const uint64_t batch_size = data.size();
    std::vector<float> inputs;
    std::vector<float> policy_teachers(POLICY_DIM * batch_size, 0.0);

    static Position pos;
    for (uint64_t i = 0; i < batch_size; i++) {
        //入力
        pos.fromStr(data[i].position_str);
        std::vector<float> curr_feature = pos.makeFeature();
        inputs.insert(inputs.end(), curr_feature.begin(), curr_feature.end());

        //policyの教師信号
        for (const std::pair<int32_t, float>& e : data[i].policy) {
            policy_teachers[i * POLICY_DIM + e.first] = e.second;
        }
    }

    torch::Tensor x = (fp16_ ? torch::tensor(inputs).to(device_, torch::kHalf) : torch::tensor(inputs).to(device_));
    torch::Tensor policy_logit = forward(x);
    torch::Tensor policy_teacher = torch::tensor(policy_teachers).to(device_).view({ -1, POLICY_DIM });

    std::vector<torch::Tensor> loss;

    //教師との交差エントロピー
    loss.push_back(policyLoss(policy_logit, policy_teacher));

    //エントロピー正則化
    loss.push_back(entropyLoss(policy_logit));

    return loss;
}

torch::Tensor SimpleMLPImpl::inferPolicy(const Position& pos) {
    std::vector<float> inputs = pos.makeFeature();
    torch::Tensor x = (fp16_ ? torch::tensor(inputs).to(device_, torch::kHalf) : torch::tensor(inputs).to(device_));
    return forward(x);
}

void SimpleMLPImpl::setGPU(int16_t gpu_id, bool fp16) {
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    fp16_ = fp16;
    (fp16_ ? to(device_, torch::kHalf) : to(device_, torch::kFloat));
}

torch::Tensor SimpleMLPImpl::forward(const torch::Tensor& x) {
    torch::Tensor y = x.view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
    y = encoder->forward(y);
    y = y.view({ -1, HIDDEN_DIM });
    return policy_head->forward(y);
}