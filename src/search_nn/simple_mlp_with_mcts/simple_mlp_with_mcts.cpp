#include "simple_mlp_with_mcts.hpp"
#include "../../hash_table.hpp"
#include "../../searcher.hpp"

SimpleMLPWithMCTSImpl::SimpleMLPWithMCTSImpl(const SearchOptions& search_options) : BaseModel(search_options) {}

std::vector<std::tuple<torch::Tensor, torch::Tensor>> SimpleMLPWithMCTSImpl::search(std::vector<Position>& positions) {
    if (positions.size() != 1) {
        std::cerr << "SimpleMLPWithMCTSは1局面の推論にしか対応していない" << std::endl;
        std::exit(1);
    }

    std::ofstream ofs("positions.txt", std::ios::app);
    ofs << positions[0].toStr() << std::endl;

    HashTable hash_table(search_options_.search_limit * 2);

    GPUQueue gpu_queue;
    Searcher searcher(search_options_, hash_table, gpu_queue);

    auto gpu_func = [&]() {
        if (!gpu_queue.inputs.empty()) {
            torch::Tensor x = torch::tensor(gpu_queue.inputs).to(device_);
            x = x.view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
            x = encoder_->forward(x);
            x = torch::flatten(x, 1);

            torch::Tensor policy_logit = base_policy_head_->forward(x);
            torch::Tensor value = base_value_head_->forward(x);

            //書き込み
            for (uint64_t i = 0; i < gpu_queue.indices.size(); i++) {
                std::unique_lock<std::mutex> lock(hash_table[gpu_queue.indices[i]].mutex);
                HashEntry& curr_node = hash_table[gpu_queue.indices[i]];

                uint64_t moves_num = curr_node.moves.size();
                curr_node.moves.shrink_to_fit();
                curr_node.nn_policy.resize(moves_num);
                curr_node.nn_policy.shrink_to_fit();
                curr_node.child_indices.assign(moves_num, HashTable::NOT_EXPANDED);
                curr_node.child_indices.shrink_to_fit();
                curr_node.N.assign(moves_num, 0);
                curr_node.N.shrink_to_fit();
                curr_node.virtual_N.assign(moves_num, 0);
                curr_node.virtual_N.shrink_to_fit();

                for (uint64_t j = 0; j < moves_num; j++) {
                    curr_node.nn_policy[j] = policy_logit[i][curr_node.moves[j].toLabel()].item<float>();
                }
                curr_node.nn_policy = softmax(curr_node.nn_policy);
                curr_node.value = value[i].item<float>();
                curr_node.evaled = true;
            }
        }
        gpu_queue.inputs.clear();
        gpu_queue.hash_tables.clear();
        gpu_queue.indices.clear();
    };

    std::stack<Index> dummy;
    std::stack<int32_t> dummy2;
    hash_table.root_index = searcher.expand(positions[0], dummy, dummy2);
    gpu_func();
    searcher.clearBackupQueue();

    eval();
    torch::NoGradGuard no_grad_guard;

    HashEntry& root_node = hash_table[hash_table.root_index];

    if (root_node.moves.size() == 1) {
        root_node.N[0] = search_options_.search_limit;
    } else {
        for (int64_t _ = 0; _ < search_options_.search_limit; _++) {
            //選択
            searcher.select(positions[0]);

            //展開
            gpu_func();

            //バックアップ
            searcher.backupAll();
        }
    }

    std::vector<std::tuple<torch::Tensor, torch::Tensor>> result;
    torch::Tensor policy = torch::zeros({ 1, 1, POLICY_DIM });
    torch::Tensor value;

    for (uint64_t i = 0; i < root_node.moves.size(); i++) {
        policy[0][0][root_node.moves[i].toLabel()] = root_node.N[i];
    }

    result.emplace_back(policy, value);
    return result;
}