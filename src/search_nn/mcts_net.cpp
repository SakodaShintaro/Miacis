#include "mcts_net.hpp"
#include <stack>

MCTSNet::MCTSNet(const SearchOptions& search_options) : search_options_(search_options),
                                                        hash_table_(search_options.USI_Hash * 1024 * 1024 / 10000) {}

Move MCTSNet::think(Position& root, int64_t time_limit) {
    //思考を行う
    //時間制限、あるいはノード数制限に基づいて何回やるかを決める

    //ルートノードについての設定
    hash_table_.deleteOldHash(root, search_options_.leave_root);
    hash_table_.root_index = hash_table_.findSameHashIndex(root);
    if (hash_table_.root_index == (Index)hash_table_.size()) {
        hash_table_.root_index = hash_table_.searchEmptyIndex(root);
    }
    HashEntryForMCTSNet& root_entry = hash_table_[hash_table_.root_index];
    root_entry.embedding_vector = neural_networks_->embed(root.makeFeature());

    for (int64_t _ = 0; _ < search_options_.search_limit; _++) {
        //1回探索する

        //到達したノードの履歴
        std::stack<Index> indices;

        //(1)選択
        while (true) {
            Index index = hash_table_.findSameHashIndex(root);
            if (index == (Index)hash_table_.size()) {
                //未展開のノードだったら次にここを評価
                break;
            } else {
                const HashEntryForMCTSNet& entry = hash_table_[index];
                torch::Tensor h = entry.embedding_vector;
                torch::Tensor policy_logit = neural_networks_->simulationPolicy(h);

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
        torch::Tensor h = (hash_table_[index].embedding_vector = neural_networks_->embed(feature));

        //(3)バックアップ
        while (!indices.empty()) {
            Index top = indices.top();
            indices.pop();

            //Backup Networkにより更新(差分更新)
            h = (hash_table_[top].embedding_vector = neural_networks_->backup(hash_table_[top].embedding_vector, h));

            root.undo();
        }
    }

    //最終的な行動決定
    //Readout Networkにより最終決定
    torch::Tensor h = root_entry.embedding_vector;
    torch::Tensor policy_logit = neural_networks_->readoutPolicy(h);

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