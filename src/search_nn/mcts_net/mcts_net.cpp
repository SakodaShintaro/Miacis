#include "mcts_net.hpp"
#include <stack>

const std::string MCTSNetImpl::MODEL_PREFIX = "mcts_net";
const std::string MCTSNetImpl::DEFAULT_MODEL_NAME = MCTSNetImpl::MODEL_PREFIX + ".model";

static const float LOG_SOFTMAX_THRESHOLD = std::log(1.0 / POLICY_DIM);

MCTSNetImpl::MCTSNetImpl(const SearchOptions& search_options)
    : search_options_(search_options),
      hash_table_(std::min(search_options.USI_Hash * 1024 * 1024 / 10000, search_options.search_limit * 10)),
      device_(torch::kCUDA), fp16_(false), freeze_encoder_(true), gamma_(1.0) {
    constexpr int64_t HIDDEN_DIM = BOARD_WIDTH * BOARD_WIDTH * StateEncoderImpl::LAST_CHANNEL_NUM;
    simulation_policy_ =
        register_module("simulation_policy_", torch::nn::Linear(torch::nn::LinearOptions(HIDDEN_DIM, POLICY_DIM)));
    encoder_ = register_module("encoder_", StateEncoder());

    value_head_ = register_module("value_head_", torch::nn::Linear(HIDDEN_DIM, 1));

    backup_update_ = register_module("backup_update_", torch::nn::Linear(torch::nn::LinearOptions(HIDDEN_DIM * 2, HIDDEN_DIM)));
    backup_gate_ = register_module("backup_gate_", torch::nn::Linear(torch::nn::LinearOptions(HIDDEN_DIM * 2, HIDDEN_DIM)));

    readout_policy_ = register_module("readout_policy_", torch::nn::Linear(torch::nn::LinearOptions(HIDDEN_DIM, POLICY_DIM)));
}

torch::Tensor MCTSNetImpl::backup(const torch::Tensor& h1, const torch::Tensor& h2) {
    torch::Tensor cat_h = torch::cat({ h1, h2 }, 1);
    torch::Tensor gate = torch::sigmoid(backup_gate_->forward(cat_h));
    return h1 + gate * backup_update_->forward(cat_h);
}

Move MCTSNetImpl::think(Position& root, int64_t time_limit) {
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
    std::vector<torch::Tensor> root_hs;
    root_hs.push_back(root_entry.embedding_vector.to(device_));

    for (int64_t m = 1; m <= search_options_.search_limit; m++) {
        //1回探索する

        //到達したノードの履歴
        std::stack<Index> indices;

        //(1)選択
        while (true) {
            Index index = hash_table_.findSameHashIndex(root);
            if (index == (Index)hash_table_.size() || root.isFinish(score)) {
                //未展開のノードだったら次にここを評価
                break;
            } else {
                indices.push(index);

                const HashEntryForMCTSNet& entry = hash_table_[index];
                torch::Tensor h = entry.embedding_vector.to(device_);
                torch::Tensor policy_logit =
                    (search_options_.use_readout_only ? readout_policy_->forward(h) : simulation_policy_->forward(h));

                //合法手だけマスクをかける
                std::vector<Move> moves = root.generateAllMoves();
                std::vector<float> logits;
                for (const Move& move : moves) {
                    logits.push_back(policy_logit[0][move.toLabel()].item<float>());
                }
                std::vector<float> masked_policy = softmax(logits, 1.0f);
                int32_t move_id = randomChoose(masked_policy);
                root.doMove(moves[move_id]);
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

        //ルートノードの現状態を保存しておく
        root_hs.push_back(hash_table_[hash_table_.root_index].embedding_vector.to(device_));
    }

    //最終的な行動決定
    //Readout Networkにより最終決定
    torch::Tensor h = root_entry.embedding_vector.to(device_);
    torch::Tensor policy_logit = readout_policy_->forward(h);

    //合法手だけマスクをかける
    std::vector<Move> moves = root.generateAllMoves();
    std::vector<float> logits;
    for (const Move& move : moves) {
        logits.push_back(policy_logit[0][move.toLabel()].item<float>());
    }

    if (search_options_.search_limit > 0 && search_options_.print_policy_num > 0) {
        torch::Tensor policy_logit0 = readout_policy_->forward(root_hs.front());
        std::vector<float> logits0;
        for (const Move& move : moves) {
            logits0.push_back(policy_logit0[0][move.toLabel()].item<float>());
        }

        std::vector<float> masked_policy = softmax(logits, 1.0f);
        std::vector<float> masked_policy0 = softmax(logits0, 1.0f);

        struct MoveWithInfo {
            Move move;
            float policy, policy0;
            bool operator<(const MoveWithInfo& rhs) const { return (policy - policy0) < (rhs.policy - rhs.policy0); }
        };

        std::vector<MoveWithInfo> move_with_info(moves.size());
        for (uint64_t i = 0; i < moves.size(); i++) {
            move_with_info[i].move = moves[i];
            move_with_info[i].policy0 = masked_policy0[i];
            move_with_info[i].policy = masked_policy[i];
        }
        std::sort(move_with_info.begin(), move_with_info.end());

        std::cout << std::fixed;
        for (uint64_t i = std::max((int64_t)0, (int64_t)moves.size() - search_options_.print_policy_num); i < moves.size(); i++) {
            const MoveWithInfo& m = move_with_info[i];
            std::cout << "info string " << m.policy0 << "       " << m.policy << "       " << m.move.toPrettyStr() << std::endl;
        }
        std::cout << "info string 探索前Policy   " << search_options_.search_limit << "回探索後Policy" << std::endl;
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

std::vector<torch::Tensor> MCTSNetImpl::loss(const std::vector<LearningData>& data, bool use_policy_gradient) {
    std::cerr << "No Implementation" << std::endl;
    std::exit(1);
}

void MCTSNetImpl::setGPU(int16_t gpu_id, bool fp16) {
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    fp16_ = fp16;
    (fp16_ ? to(device_, torch::kHalf) : to(device_, torch::kFloat));
}

void MCTSNetImpl::loadPretrain(const std::string& encoder_path, const std::string& policy_head_path,
                               const std::string& value_head_path) {
    std::ifstream encoder_file(encoder_path);
    if (encoder_file.is_open()) {
        torch::load(encoder_, encoder_path);
    }
    std::ifstream policy_head_file(policy_head_path);
    if (policy_head_file.is_open()) {
        torch::load(simulation_policy_, policy_head_path);
        torch::load(readout_policy_, policy_head_path);
    }
    std::ifstream value_head_file(value_head_path);
    if (value_head_file.is_open()) {
        torch::load(value_head_, value_head_path);
    }
}

void MCTSNetImpl::setOption(bool freeze_encoder, float gamma) {
    freeze_encoder_ = freeze_encoder;
    gamma_ = gamma;
}