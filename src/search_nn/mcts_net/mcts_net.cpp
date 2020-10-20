#include "mcts_net.hpp"
#include "../common.hpp"
#include <stack>

MCTSNetImpl::MCTSNetImpl(const SearchOptions& search_options)
    : BaseModel(search_options),
      hash_table_(std::min(search_options.USI_Hash * 1024 * 1024 / 10000, search_options.search_limit * 10)) {
    constexpr int64_t HIDDEN_DIM = BOARD_WIDTH * BOARD_WIDTH * StateEncoderImpl::LAST_CHANNEL_NUM;

    backup_update_ = register_module("backup_update_", torch::nn::Linear(torch::nn::LinearOptions(HIDDEN_DIM * 2, HIDDEN_DIM)));
    backup_gate_ = register_module("backup_gate_", torch::nn::Linear(torch::nn::LinearOptions(HIDDEN_DIM * 2, HIDDEN_DIM)));

    readout_policy_ = register_module("readout_policy_", torch::nn::Linear(torch::nn::LinearOptions(HIDDEN_DIM, POLICY_DIM)));
}

torch::Tensor MCTSNetImpl::backup(const torch::Tensor& h1, const torch::Tensor& h2) {
    torch::Tensor cat_h = torch::cat({ h1, h2 }, 1);
    torch::Tensor gate = torch::sigmoid(backup_gate_->forward(cat_h));
    return h1 + gate * backup_update_->forward(cat_h);
}

//Move MCTSNetImpl::think(Position& root, int64_t time_limit) {
//    //思考を行う
//    //時間制限、あるいはノード数制限に基づいて何回やるかを決める
//    //合法手が0だったら投了
//    float score{};
//    if (root.isFinish(score) && score == MIN_SCORE) {
//        return NULL_MOVE;
//    }
//
//    //バッチ化している関数と共通化している都合上、盤面をvector化
//    std::vector<Position> positions;
//    positions.push_back(root);
//
//    //出力方策の系列を取得
//    std::vector<torch::Tensor> policy_logits = search(positions);
//
//    //合法手だけマスクをかける
//    std::vector<Move> moves = root.generateAllMoves();
//    std::vector<float> logits;
//    for (const Move& move : moves) {
//        logits.push_back(policy_logits.back()[0][0][move.toLabel()].item<float>());
//    }
//
//    if (root.turnNumber() <= search_options_.random_turn) {
//        //Softmaxの確率に従って選択
//        std::vector<float> masked_policy = softmax(logits, 1.0f);
//        int32_t move_id = randomChoose(masked_policy);
//        return moves[move_id];
//    } else {
//        //最大のlogitを持つ行動を選択
//        int32_t move_id = std::max_element(logits.begin(), logits.end()) - logits.begin();
//        return moves[move_id];
//    }
//
//
//    //ルートノードについての設定
//    hash_table_.deleteOldHash();
//    hash_table_.root_index = hash_table_.findSameHashIndex(root);
//    if (hash_table_.root_index == (Index)hash_table_.size()) {
//        hash_table_.root_index = hash_table_.searchEmptyIndex(root);
//    }
//    HashEntryForMCTSNet& root_entry = hash_table_[hash_table_.root_index];
//    root_entry.embedding_vector = encoder_->embed(root.makeFeature(), device_, fp16_, freeze_encoder_).cpu();
//    std::vector<torch::Tensor> root_hs;
//    root_hs.push_back(root_entry.embedding_vector.to(device_));
//
//    for (int64_t m = 1; m <= search_options_.search_limit; m++) {
//        //1回探索する
//
//        //到達したノードの履歴
//        std::stack<Index> indices;
//
//        //(1)選択
//        while (true) {
//            Index index = hash_table_.findSameHashIndex(root);
//            if (index == (Index)hash_table_.size() || root.isFinish(score)) {
//                //未展開のノードだったら次にここを評価
//                break;
//            } else {
//                indices.push(index);
//
//                const HashEntryForMCTSNet& entry = hash_table_[index];
//                torch::Tensor h = entry.embedding_vector.to(device_);
//                torch::Tensor policy_logit = sim_policy_head_->forward(h);
//
//                //合法手だけマスクをかける
//                std::vector<Move> moves = root.generateAllMoves();
//                std::vector<float> logits;
//                for (const Move& move : moves) {
//                    logits.push_back(policy_logit[0][move.toLabel()].item<float>());
//                }
//                std::vector<float> masked_policy = softmax(logits, 1.0f);
//                int32_t move_id = randomChoose(masked_policy);
//                root.doMove(moves[move_id]);
//            }
//        }
//
//        //(2)評価
//        std::vector<float> feature = root.makeFeature();
//        Index index = hash_table_.searchEmptyIndex(root);
//        torch::Tensor h = encoder_->embed(feature, device_, fp16_, freeze_encoder_);
//        hash_table_[index].embedding_vector = h.cpu();
//
//        //(3)バックアップ
//        while (!indices.empty()) {
//            Index top = indices.top();
//            indices.pop();
//
//            //Backup Networkにより更新(差分更新)
//            h = backup(hash_table_[top].embedding_vector.to(device_), h);
//            hash_table_[top].embedding_vector = h.cpu();
//
//            root.undo();
//        }
//
//        //ルートノードの現状態を保存しておく
//        root_hs.push_back(hash_table_[hash_table_.root_index].embedding_vector.to(device_));
//    }
//
//    //最終的な行動決定
//    //Readout Networkにより最終決定
//    torch::Tensor h = root_entry.embedding_vector.to(device_);
//    torch::Tensor policy_logit = readout_policy_->forward(h);
//
//    //合法手だけマスクをかける
//    std::vector<Move> moves = root.generateAllMoves();
//    std::vector<float> logits;
//    for (const Move& move : moves) {
//        logits.push_back(policy_logit[0][move.toLabel()].item<float>());
//    }
//
//    if (search_options_.search_limit > 0 && search_options_.print_policy_num > 0) {
//        torch::Tensor policy_logit0 = readout_policy_->forward(root_hs.front());
//        std::vector<float> logits0;
//        for (const Move& move : moves) {
//            logits0.push_back(policy_logit0[0][move.toLabel()].item<float>());
//        }
//
//        std::vector<float> masked_policy = softmax(logits, 1.0f);
//        std::vector<float> masked_policy0 = softmax(logits0, 1.0f);
//
//        struct MoveWithInfo {
//            Move move;
//            float policy, policy0;
//            bool operator<(const MoveWithInfo& rhs) const { return (policy - policy0) < (rhs.policy - rhs.policy0); }
//        };
//
//        std::vector<MoveWithInfo> move_with_info(moves.size());
//        for (uint64_t i = 0; i < moves.size(); i++) {
//            move_with_info[i].move = moves[i];
//            move_with_info[i].policy0 = masked_policy0[i];
//            move_with_info[i].policy = masked_policy[i];
//        }
//        std::sort(move_with_info.begin(), move_with_info.end());
//
//        std::cout << std::fixed;
//        for (uint64_t i = std::max((int64_t)0, (int64_t)moves.size() - search_options_.print_policy_num); i < moves.size(); i++) {
//            const MoveWithInfo& m = move_with_info[i];
//            std::cout << "info string " << m.policy0 << "       " << m.policy << "       " << m.move.toPrettyStr() << std::endl;
//        }
//        std::cout << "info string 探索前Policy   " << search_options_.search_limit << "回探索後Policy" << std::endl;
//    }
//
//    if (root.turnNumber() <= search_options_.random_turn) {
//        //Softmaxの確率に従って選択
//        std::vector<float> masked_policy = softmax(logits, 1.0f);
//        int32_t move_id = randomChoose(masked_policy);
//        return moves[move_id];
//    } else {
//        //最大のlogitを持つ行動を選択
//        int32_t move_id = std::max_element(logits.begin(), logits.end()) - logits.begin();
//        return moves[move_id];
//    }
//}

std::vector<torch::Tensor> MCTSNetImpl::search(std::vector<Position>& positions) {
    //バッチサイズを取得しておく
    const uint64_t batch_size = positions.size();

    //探索回数
    const int64_t M = search_options_.search_limit;

    //置換表を準備
    std::vector<HashTableForMCTSNet> hash_tables(batch_size, HashTableForMCTSNet(M * 10));

    //盤面を復元
    std::vector<float> root_features;
    for (uint64_t i = 0; i < batch_size; i++) {
        hash_tables[i].root_index = hash_tables[i].findSameHashIndex(positions[i]);
        if (hash_tables[i].root_index == (Index)hash_tables[i].size()) {
            hash_tables[i].root_index = hash_tables[i].searchEmptyIndex(positions[i]);
        }

        std::vector<float> f = positions[i].makeFeature();
        root_features.insert(root_features.end(), f.begin(), f.end());
    }

    //GPUで計算
    torch::Tensor root_embed = encoder_->embed(root_features, device_, fp16_, freeze_encoder_);

    //各探索後のpolicy_logits
    std::vector<torch::Tensor> policy_logits;

    //0回目
    policy_logits.push_back(readout_policy_->forward(root_embed).view({ 1, (int64_t)batch_size, POLICY_DIM }));

    //0回目の情報
    for (uint64_t i = 0; i < batch_size; i++) {
        hash_tables[i][hash_tables[i].root_index].embedding_vector = root_embed[i];
    }

    //探索の履歴
    std::vector<std::stack<Index>> indices(batch_size);

    //規定回数探索
    for (int64_t m = 1; m <= M; m++) {
        //(1)select
        while (true) {
            //今回の評価要求
            std::vector<torch::Tensor> embedding_vectors;
            std::vector<uint64_t> ids;

            //評価要求を貯める
            for (uint64_t i = 0; i < batch_size; i++) {
                //各局面でselectをしていく
                Position& pos = positions[i];

                Index index = hash_tables[i].findSameHashIndex(pos);
                float score{};
                if (index == (Index)hash_tables[i].size() || pos.isFinish(score)) {
                    //リーフノードまで達しているのでスキップ
                    continue;
                }

                indices[i].push(index);

                const HashEntryForMCTSNet& entry = hash_tables[i][index];
                torch::Tensor h = entry.embedding_vector.to(device_);

                //評価要求に貯める
                embedding_vectors.push_back(h);
                ids.push_back(i);
            }

            //全ての局面がリーフノードに達していたら終わり
            if (embedding_vectors.empty()) {
                break;
            }

            //GPUで計算
            torch::Tensor h = torch::stack(embedding_vectors);
            torch::Tensor policy_logit = sim_policy_head_->forward(h);

            //計算結果を反映
            for (uint64_t j = 0; j < ids.size(); j++) {
                uint64_t i = ids[j];

                //合法手だけマスクをかける
                std::vector<Move> moves = positions[i].generateAllMoves();
                std::vector<float> logits;
                for (const Move& move : moves) {
                    logits.push_back(policy_logit[j][move.toLabel()].item<float>());
                }
                std::vector<float> masked_policy = softmax(logits, 1.0f);
                int32_t move_id = randomChoose(masked_policy);
                positions[i].doMove(moves[move_id]);
            }
        }

        //(2)評価
        std::vector<float> features;
        for (uint64_t i = 0; i < batch_size; i++) {
            std::vector<float> f = positions[i].makeFeature();
            features.insert(features.end(), f.begin(), f.end());
        }
        torch::Tensor h = encoder_->embed(features, device_, fp16_, freeze_encoder_);
        for (uint64_t i = 0; i < batch_size; i++) {
            std::vector<float> f = positions[i].makeFeature();
            features.insert(features.end(), f.begin(), f.end());
            Index index = hash_tables[i].searchEmptyIndex(positions[i]);
            hash_tables[i][index].embedding_vector = h[i].cpu();
        }

        std::vector<Index> next_indices(batch_size, -1);

        //(3)バックアップ
        while (true) {
            std::vector<torch::Tensor> left, right;
            std::vector<uint64_t> ids;
            std::vector<Index> update_indices;

            for (uint64_t i = 0; i < batch_size; i++) {
                if (indices[i].empty()) {
                    //もうbackupは終わった
                    continue;
                }

                Index top = indices[i].top();
                indices[i].pop();
                update_indices.push_back(top);
                left.push_back(hash_tables[i][top].embedding_vector.to(device_));
                right.push_back(next_indices[i] == -1 ? h[i] : hash_tables[i][next_indices[i]].embedding_vector.to(device_));
                ids.push_back(i);
            }

            if (ids.empty()) {
                break;
            }

            //GPUで計算
            torch::Tensor h1 = torch::stack(left);  //[backup中のもの, HIDDEN_DIM]
            torch::Tensor h2 = torch::stack(right); //[backup中のもの, HIDDEN_DIM]
            torch::Tensor backup_h = backup(h1, h2);

            for (uint64_t j = 0; j < ids.size(); j++) {
                uint64_t i = ids[j];
                hash_tables[i][update_indices[j]].embedding_vector = backup_h[j];
                next_indices[i] = update_indices[j];
                positions[i].undo();
            }
        }

        std::vector<torch::Tensor> root_hs;
        for (uint64_t i = 0; i < batch_size; i++) {
            root_hs.push_back(hash_tables[i][hash_tables[i].root_index].embedding_vector.to(device_));
        }
        torch::Tensor root_h = torch::stack(root_hs);
        policy_logits.push_back(readout_policy_->forward(root_h).view({ 1, (int64_t)batch_size, POLICY_DIM }));
    }

    return policy_logits;
}