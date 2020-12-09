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

    //GPUで最初の埋め込みを計算
    torch::Tensor root_embed = embed(positions)[0];

    //各探索後のpolicy_logits
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> policy_and_value;

    //0回目
    if (!last_only_ || search_options_.search_limit == 0) {
        policy_and_value.emplace_back(readout_policy_->forward(root_embed).view({ 1, (int64_t)batch_size, POLICY_DIM }),
                                      torch::tanh(readout_value_->forward(root_embed)));
    }

    auto save = [&](const torch::Tensor& embed_vector, bool root) {
        torch::Tensor policy_logit = base_policy_head_->forward(embed_vector).cpu();

        //埋め込みを置換表に保存
        for (uint64_t i = 0; i < batch_size; i++) {
            HashTableForMCTSNet& table = hash_tables[i];
            Index index = table.searchEmptyIndex(positions[i]);
            if (root) {
                table.root_index = index;
            }

            HashEntryForMCTSNet& entry = table[index];

            entry.embedding_vector = embed_vector[i];
            entry.moves = positions[i].generateAllMoves();
            entry.nn_policy.resize(entry.moves.size());
            for (uint64_t j = 0; j < entry.moves.size(); j++) {
                entry.nn_policy[j] = policy_logit[i][entry.moves[j].toLabel()].item<float>();
            }
            entry.nn_policy = softmax(entry.nn_policy);
        }
    };

    //埋め込みを置換表に保存
    save(root_embed, true);

    //探索の履歴
    std::vector<std::stack<Index>> indices(batch_size);

    //規定回数探索
    for (int64_t m = 1; m <= M; m++) {
        //(1)select
        //各局面で置換表に登録されていない状態まで降りる
        for (uint64_t i = 0; i < batch_size; i++) {
            Index index = hash_tables[i].findSameHashIndex(positions[i]);
            while (index != hash_tables[i].size()) {
                HashEntryForMCTSNet& curr_entry = hash_tables[i][index];
                indices[i].push(index);
                int32_t move_id = randomChoose(curr_entry.nn_policy);
                positions[i].doMove(curr_entry.moves[move_id]);
                index = hash_tables[i].findSameHashIndex(positions[i]);
            }
        }

        //(2)評価
        torch::Tensor h = embed(positions)[0];
        save(h, false);

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
        if (!last_only_ || m == search_options_.search_limit) {
            policy_and_value.emplace_back(readout_policy_->forward(root_h).view({ 1, (int64_t)batch_size, POLICY_DIM }),
                                          torch::tanh(readout_value_->forward(root_h)));
        }
    }

    return policy_and_value;
}