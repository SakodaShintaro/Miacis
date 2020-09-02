#include "mcts_net.hpp"
#include "../common.hpp"
#include <stack>

const std::string MCTSNetImpl::MODEL_PREFIX = "mcts_net";
const std::string MCTSNetImpl::DEFAULT_MODEL_NAME = MCTSNetImpl::MODEL_PREFIX + ".model";

MCTSNetImpl::MCTSNetImpl(const SearchOptions& search_options)
    : search_options_(search_options),
      hash_table_(std::min(search_options.USI_Hash * 1024 * 1024 / 10000, search_options.search_limit * 10)),
      device_(torch::kCUDA), fp16_(false) {
    constexpr int64_t HIDDEN_DIM = BOARD_WIDTH * BOARD_WIDTH * StateEncoderImpl::LAST_CHANNEL_NUM;
    simulation_policy_ =
        register_module("simulation_policy_", torch::nn::Linear(torch::nn::LinearOptions(HIDDEN_DIM, POLICY_DIM)));
    encoder_ = register_module("encoder_", StateEncoder());

    backup_update_ = register_module("backup_update_", torch::nn::Linear(torch::nn::LinearOptions(HIDDEN_DIM * 2, HIDDEN_DIM)));
    backup_gate_ = register_module("backup_gate_", torch::nn::Linear(torch::nn::LinearOptions(HIDDEN_DIM * 2, HIDDEN_DIM)));

    readout_policy_ = register_module("readout_policy_", torch::nn::Linear(torch::nn::LinearOptions(HIDDEN_DIM, POLICY_DIM)));
}

torch::Tensor MCTSNetImpl::simulationPolicy(const torch::Tensor& h) { return simulation_policy_->forward(h); }

torch::Tensor MCTSNetImpl::backup(const torch::Tensor& h1, const torch::Tensor& h2) {
    torch::Tensor cat_h = torch::cat({ h1, h2 }, 1);
    torch::Tensor gate = torch::sigmoid(backup_gate_->forward(cat_h));
    return h1 + gate * backup_update_->forward(cat_h);
}

torch::Tensor MCTSNetImpl::readoutPolicy(const torch::Tensor& h) { return readout_policy_->forward(h); }

Move MCTSNetImpl::think(Position& root, int64_t time_limit, bool save_info_to_learn) {
    //思考を行う
    //時間制限、あるいはノード数制限に基づいて何回やるかを決める
    //合法手が0だったら投了
    float score{};
    if (root.isFinish(score) && score == MIN_SCORE) {
        return NULL_MOVE;
    }

    //ルートノードについての設定
    hash_table_.deleteOldHash();
    hash_table_.root_index = hash_table_.findSameHashIndex(root);
    if (hash_table_.root_index == (Index)hash_table_.size()) {
        hash_table_.root_index = hash_table_.searchEmptyIndex(root);
    }
    HashEntryForMCTSNet& root_entry = hash_table_[hash_table_.root_index];
    root_entry.embedding_vector = encoder_->embed(root.makeFeature(), device_, fp16_, freeze_encoder_).cpu();
    root_h_.push_back(root_entry.embedding_vector.to(device_));

    //0回目
    if (save_info_to_learn) {
        probs_.push_back(torch::ones({ 1 }).to(device_));
    }

    for (int64_t m = 1; m <= search_options_.search_limit; m++) {
        //1回探索する

        //到達したノードの履歴
        std::stack<Index> indices;

        //(1)選択
        if (save_info_to_learn) {
            probs_.push_back(torch::ones({ 1 }).to(device_));
        }
        while (true) {
            Index index = hash_table_.findSameHashIndex(root);
            float score{};
            if (index == (Index)hash_table_.size() || root.isFinish(score)) {
                //未展開のノードだったら次にここを評価
                break;
            } else {
                indices.push(index);

                const HashEntryForMCTSNet& entry = hash_table_[index];
                torch::Tensor h = entry.embedding_vector.to(device_);
                torch::Tensor policy_logit = simulationPolicy(h);

                //合法手だけマスクをかける
                std::vector<Move> moves = root.generateAllMoves();
                std::vector<float> logits;
                for (const Move& move : moves) {
                    logits.push_back(policy_logit[0][move.toLabel()].item<float>());
                }
                std::vector<float> masked_policy = softmax(logits, 1.0f);
                int32_t move_id = randomChoose(masked_policy);
                root.doMove(moves[move_id]);
                if (save_info_to_learn) {
                    torch::Tensor policy = torch::softmax(policy_logit, 1);
                    probs_[m] *= policy[0][moves[move_id].toLabel()];
                }
            }
        }

        //(2)評価
        std::vector<float> feature = root.makeFeature();
        Index index = hash_table_.searchEmptyIndex(root);
        torch::Tensor h = encoder_->embed(feature, device_, fp16_, freeze_encoder_);
        hash_table_[index].embedding_vector = h.cpu();

        //(3)バックアップ
        while (!indices.empty()) {
            Index top = indices.top();
            indices.pop();

            //Backup Networkにより更新(差分更新)
            h = backup(hash_table_[top].embedding_vector.to(device_), h);
            hash_table_[top].embedding_vector = h.cpu();

            root.undo();
        }

        if (save_info_to_learn) {
            //ルートノードの現状態を保存しておく
            root_h_.push_back(hash_table_[hash_table_.root_index].embedding_vector.to(device_));
        }
    }

    //最終的な行動決定
    //Readout Networkにより最終決定
    torch::Tensor h = root_entry.embedding_vector.to(device_);
    torch::Tensor policy_logit = readoutPolicy(h);

    //合法手だけマスクをかける
    std::vector<Move> moves = root.generateAllMoves();
    std::vector<float> logits;
    for (const Move& move : moves) {
        logits.push_back(policy_logit[0][move.toLabel()].item<float>());
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

std::vector<torch::Tensor> MCTSNetImpl::loss(const std::vector<LearningData>& data, bool freeze_encoder) {
    //現状バッチサイズは1のみに対応
    assert(data.size() == 1);

    //設定を内部の変数に格納
    freeze_encoder_ = freeze_encoder;

    Position root;
    root.fromStr(data.front().position_str);

    //探索を行い、途中のルート埋め込みベクトル,各探索の確率等を保存しておく
    root_h_.clear();
    probs_.clear();
    think(root, INT_MAX, true);

    assert(root_h_.size() == probs_.size());

    const int64_t M = root_h_.size() - 1;

    //policyの教師信号
    torch::Tensor policy_teacher = getPolicyTeacher(data, device_);

    //各探索後の損失を計算
    std::vector<torch::Tensor> l(M + 1);
    for (int64_t m = 0; m <= M; m++) {
        torch::Tensor policy_logit = readoutPolicy(root_h_[m]);
        torch::Tensor log_softmax = torch::log_softmax(policy_logit, 1);
        torch::Tensor clipped = torch::clamp_min(log_softmax, -20);
        l[m] = (-policy_teacher * clipped).sum();
    }

    //損失の差分を計算
    std::vector<torch::Tensor> r(M + 1);
    for (int64_t m = 1; m <= M; m++) {
        r[m] = l[m - 1] - l[m];
    }

    //重み付き累積和
    constexpr float gamma = 1.0;
    std::vector<torch::Tensor> R(M + 1);
    R[M] = r[M].detach().to(device_);
    for (int64_t m = M - 1; m >= 1; m--) {
        //逆順に求めていくことでO(M)
        R[m] = (r[m] + gamma * R[m + 1]).detach().to(device_);
    }

    std::vector<torch::Tensor> loss;
    loss.push_back(l[M].view({ 1 }));
    for (int64_t m = 1; m <= M; m++) {
        loss.push_back(-probs_[m] * R[m]);
    }

    return loss;
}

std::vector<torch::Tensor> MCTSNetImpl::validationLoss(const std::vector<LearningData>& data) {
    //現状バッチサイズは1のみに対応
    assert(data.size() == 1);

    Position root;
    root.fromStr(data.front().position_str);

    //探索を行い、途中のルート埋め込みベクトル,各探索の確率等を保存しておく
    root_h_.clear();
    probs_.clear();
    think(root, INT_MAX, true);

    assert(root_h_.size() == probs_.size());

    const int64_t M = root_h_.size();

    //policyの教師信号
    torch::Tensor policy_teacher = getPolicyTeacher(data, device_);

    //各探索後の損失を計算
    std::vector<torch::Tensor> loss;
    for (int64_t m = 0; m < M; m++) {
        torch::Tensor policy_logit = readoutPolicy(root_h_[m]);
        torch::Tensor log_softmax = torch::log_softmax(policy_logit, 1);
        torch::Tensor clipped = torch::clamp_min(log_softmax, -20);
        torch::Tensor curr_loss = (-policy_teacher * clipped).sum().view({ 1 });
        loss.push_back(curr_loss);
    }

    return loss;
}

void MCTSNetImpl::setGPU(int16_t gpu_id, bool fp16) {
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    fp16_ = fp16;
    (fp16_ ? to(device_, torch::kHalf) : to(device_, torch::kFloat));
}

void MCTSNetImpl::loadPretrain(const std::string& encoder_path, const std::string& policy_head_path) {
    torch::load(encoder_, encoder_path);
    torch::load(simulation_policy_, policy_head_path);
    torch::load(readout_policy_, policy_head_path);
}