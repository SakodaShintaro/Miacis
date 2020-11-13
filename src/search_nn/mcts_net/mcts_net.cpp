#include "mcts_net.hpp"
#include <stack>

MCTSNetImpl::MCTSNetImpl(const SearchOptions& search_options)
    : BaseModel(search_options),
      hash_table_(std::min(search_options.USI_Hash * 1024 * 1024 / 10000, search_options.search_limit * 10)) {
    constexpr int64_t HIDDEN_DIM = BOARD_WIDTH * BOARD_WIDTH * StateEncoderImpl::LAST_CHANNEL_NUM;

    backup_update_ = register_module("backup_update_", torch::nn::Linear(torch::nn::LinearOptions(HIDDEN_DIM * 2, HIDDEN_DIM)));
    backup_gate_ = register_module("backup_gate_", torch::nn::Linear(torch::nn::LinearOptions(HIDDEN_DIM * 2, HIDDEN_DIM)));

    readout_policy_ = register_module("readout_policy_", torch::nn::Linear(torch::nn::LinearOptions(HIDDEN_DIM, POLICY_DIM)));
    readout_value_ = register_module("readout_value_", torch::nn::Linear(torch::nn::LinearOptions(HIDDEN_DIM, 1)));
}

torch::Tensor MCTSNetImpl::backup(const torch::Tensor& h1, const torch::Tensor& h2) {
    torch::Tensor cat_h = torch::cat({ h1, h2 }, 1);
    torch::Tensor gate = torch::sigmoid(backup_gate_->forward(cat_h));
    return h1 + gate * backup_update_->forward(cat_h);
}

std::vector<std::tuple<torch::Tensor, torch::Tensor>> MCTSNetImpl::search(std::vector<Position>& positions) {
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
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> policy_and_value;

    //0回目
    policy_and_value.emplace_back(readout_policy_->forward(root_embed).view({ 1, (int64_t)batch_size, POLICY_DIM }),
                                  torch::tanh(readout_value_->forward(root_embed)));

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
            torch::Tensor policy_logit = base_policy_head_->forward(h);

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
        policy_and_value.emplace_back(readout_policy_->forward(root_h).view({ 1, (int64_t)batch_size, POLICY_DIM }),
                                      torch::tanh(readout_value_->forward(root_h)));
    }

    return policy_and_value;
}