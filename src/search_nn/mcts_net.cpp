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
                indices.push(index);

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

torch::Tensor MCTSNet::loss(const LearningData& datum) {
    //探索を行い、途中のルート埋め込みベクトルを保存しておく
    //各埋め込みベクトルからReadoutを行い、損失を計算

    //h = zとなる確率を損失に含む？
    //zとなる確率は積で表せる？
    //それってめっちゃ小さくなりそうな気がするが

    //各mでmになるz_mになる確率、つまり積を求めているようにしか見えない
    //それが必要なので、m回目の探索でどの行動を選んだかの系列は必要
    //少なくとも中間表現の埋め込みベクトル系列は必要
    //そこから損失の系列 l_mを求めるところまではできる
    //そこから

    std::vector<torch::Tensor> root_h;
    std::vector<torch::Tensor> probs;
    std::vector<std::vector<Move>> selected_moves;

    Position root;
    root.fromStr(datum.position_str);

    //ルートノードについての設定
    hash_table_.deleteOldHash(root, search_options_.leave_root);
    hash_table_.root_index = hash_table_.findSameHashIndex(root);
    if (hash_table_.root_index == (Index)hash_table_.size()) {
        hash_table_.root_index = hash_table_.searchEmptyIndex(root);
    }
    HashEntryForMCTSNet& root_entry = hash_table_[hash_table_.root_index];
    root_entry.embedding_vector = neural_networks_->embed(root.makeFeature());

    for (int64_t m = 0; m < search_options_.search_limit; m++) {
        //1回探索する

        //到達したノードの履歴
        std::stack<Index> indices;

        //(1)選択
        probs[m] = torch::ones({ 1 });
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
                selected_moves[m].push_back(moves[move_id]);
                probs[m] *= masked_policy[move_id];
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

        //ルートノードの現状態を保存しておく
        root_h.push_back(hash_table_[hash_table_.root_index].embedding_vector);
    }

    torch::Tensor policy_teacher;

    //各探索後の損失を計算
    std::vector<torch::Tensor> l(search_options_.search_limit + 1);
    for (int64_t m = 0; m < search_options_.search_limit; m++) {
        l[m + 1] = (-policy_teacher * torch::log_softmax(neural_networks_->readoutPolicy(root_h[m]), 1)).sum();
    }

    //損失の差分を計算
    std::vector<torch::Tensor> r(search_options_.search_limit + 1);
    for (int64_t m = 0; m < search_options_.search_limit; m++) {
        r[m + 1] = -(l[m + 1] - l[m]);
    }

    //重み付き累積和
    constexpr float gamma = 1.0;
    std::vector<torch::Tensor> R(search_options_.search_limit + 1);
    for (int64_t m = 1; m <= search_options_.search_limit; m++) {
        R[m] = torch::zeros({ 1 });
        for (int64_t m2 = m; m2 <= search_options_.search_limit; m2++) {
            R[m] += std::pow(gamma, m2 - m) * r[m2];
        }
        R[m] = R[m].detach();
    }

    torch::Tensor loss = l[search_options_.search_limit];
    for (int64_t m = 1; m <= search_options_.search_limit; m++) {
        loss += probs[m - 1] * R[m];
    }

    return loss;
}