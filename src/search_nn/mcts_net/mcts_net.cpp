#include "mcts_net.hpp"
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

torch::Tensor MCTSNetImpl::embed(const std::vector<float>& inputs) {
    torch::Tensor x = (fp16_ ? torch::tensor(inputs).to(device_, torch::kHalf) : torch::tensor(inputs).to(device_));
    x = x.view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
    x = encoder_->forward(x);
    x = torch::flatten(x, 1);
    return x;
}

torch::Tensor MCTSNetImpl::backup(const torch::Tensor& h1, const torch::Tensor& h2) {
    torch::Tensor cat_h = torch::cat({ h1, h2 }, 1);
    torch::Tensor gate = torch::sigmoid(backup_gate_->forward(cat_h));
    return h1 + gate * backup_update_->forward(cat_h);
}

torch::Tensor MCTSNetImpl::readoutPolicy(const torch::Tensor& h) { return readout_policy_->forward(h); }

Move MCTSNetImpl::think(Position& root, int64_t time_limit, bool save_info_to_learn) {
    //思考を行う
    //時間制限、あるいはノード数制限に基づいて何回やるかを決める

    //ルートノードについての設定
    hash_table_.deleteOldHash();
    hash_table_.root_index = hash_table_.findSameHashIndex(root);
    if (hash_table_.root_index == (Index)hash_table_.size()) {
        hash_table_.root_index = hash_table_.searchEmptyIndex(root);
    }
    HashEntryForMCTSNet& root_entry = hash_table_[hash_table_.root_index];
    root_entry.embedding_vector = embed(root.makeFeature()).cpu();

    for (int64_t m = 0; m < search_options_.search_limit; m++) {
        //1回探索する

        //到達したノードの履歴
        std::stack<Index> indices;

        //(1)選択
        if (save_info_to_learn) {
            probs_.push_back(torch::ones({ 1 }).to(device_));
        }
        while (true) {
            Index index = hash_table_.findSameHashIndex(root);
            FloatType score{};
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
        torch::Tensor h = embed(feature);
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

std::vector<torch::Tensor> MCTSNetImpl::loss(const std::vector<LearningData>& data) {
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

    std::vector<float> policy_teachers(POLICY_DIM, 0.0);
    //policyの教師信号
    for (const std::pair<int32_t, float>& e : data.front().policy) {
        policy_teachers[e.first] = e.second;
    }

    torch::Tensor policy_teacher = torch::tensor(policy_teachers).to(device_).view({ -1, POLICY_DIM });

    //各探索後の損失を計算
    std::vector<torch::Tensor> l(M + 1);
    l[0] = torch::zeros({ 1 });
    std::cout << std::fixed;
    for (int64_t m = 0; m < M; m++) {
        torch::Tensor policy_logit = readoutPolicy(root_h_[m]);
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
    for (int64_t m = 1; m <= M; m++) {
        //loss.push_back(probs_[m - 1] * R[m]);
        loss.push_back(l[m].view({ 1 }));
    }

    return loss;
}

void MCTSNetImpl::setGPU(int16_t gpu_id, bool fp16) {
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    fp16_ = fp16;
    (fp16_ ? to(device_, torch::kHalf) : to(device_, torch::kFloat));
}