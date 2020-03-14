#include "searcher_for_play.hpp"
#include <thread>

SearcherForPlay::SearcherForPlay(const SearchOptions& search_options)
: stop_signal(false), search_options_(search_options), hash_table_(search_options.USI_Hash * 1024 * 1024 / 10000), mate_searcher_(hash_table_, search_options) {
    //GPUを準備
    for (int64_t i = 0; i < search_options.gpu_num; i++) {
        neural_networks_.emplace_back();
        torch::load(neural_networks_[i], search_options_.model_name);
        neural_networks_[i]->setGPU(i, search_options_.use_fp16);
        neural_networks_[i]->eval();
    }

    //GPUに対するmutexを準備
    gpu_mutexes_ = std::vector<std::mutex>(search_options.gpu_num);

    //gpu_queueとsearchersを準備
    gpu_queues_.resize(search_options.gpu_num);
    searchers_.resize(search_options.gpu_num);
    for (int64_t i = 0; i < search_options.gpu_num; i++) {
        gpu_queues_[i].resize(search_options.thread_num_per_gpu);
        for (int64_t j = 0; j < search_options.thread_num_per_gpu; j++) {
            searchers_[i].emplace_back(search_options, hash_table_, gpu_queues_[i][j]);
        }
    }

#ifdef SHOGI
    book_.open(search_options.book_file_name);
#endif
}

Move SearcherForPlay::think(Position& root, int64_t time_limit) {
    //思考開始時間をセット
    start_ = std::chrono::steady_clock::now();

#ifdef SHOGI
    if (search_options_.use_book && book_.hasEntry(root)) {
        return book_.pickOne(root, 1.0);
    }
#endif

    //制限の設定
    time_limit_ = time_limit;
    node_limit_ = search_options_.search_limit;

    //古いハッシュを削除
    hash_table_.deleteOldHash(root, search_options_.leave_root);

    //表示時刻の初期化
    next_print_time_ = search_options_.print_interval;

    //キューの初期化
    for (int64_t i = 0; i < search_options_.gpu_num; i++) {
        for (int64_t j = 0; j < search_options_.thread_num_per_gpu; j++) {
            gpu_queues_[i][j].inputs.clear();
            gpu_queues_[i][j].hash_tables.clear();
            gpu_queues_[i][j].indices.clear();
            searchers_[i][j].clearBackupQueue();
        }
    }

    //ルートノードの展開:[0][0]番目のsearcherを使う。[0][0]番目のキューに溜まる
    std::stack<Index> dummy;
    std::stack<int32_t> dummy2;
    hash_table_.root_index = searchers_[0][0].expand(root, dummy, dummy2);
    HashEntry& curr_node = hash_table_[hash_table_.root_index];

    //合法手が0だったら投了
    if (curr_node.moves.empty()) {
        return NULL_MOVE;
    }

    //GPUで計算
    if (curr_node.nn_policy.size() != curr_node.moves.size()) {
        if (gpu_queues_[0][0].inputs.empty()) {
            std::vector<FloatType> feature = root.makeFeature();
            gpu_queues_[0][0].inputs.insert(gpu_queues_[0][0].inputs.begin(), feature.begin(), feature.end());
        }
        std::pair<std::vector<PolicyType>, std::vector<ValueType>> y = neural_networks_[0]->policyAndValueBatch(gpu_queues_[0][0].inputs);

        //ルートノードへ書き込み
        curr_node.nn_policy.resize(curr_node.moves.size());
        for (uint64_t i = 0; i < curr_node.moves.size(); i++) {
            curr_node.nn_policy[i] = y.first[0][curr_node.moves[i].toLabel()];
        }
        curr_node.nn_policy = softmax(curr_node.nn_policy);
        curr_node.value = y.second[0];
        curr_node.evaled = true;
    }

    //GPUに付随するスレッドを立ち上げ
    std::vector<std::thread> threads(search_options_.gpu_num);
    for (int64_t i = 0; i < search_options_.gpu_num; i++) {
        threads[i] = std::thread(&SearcherForPlay::gpuThreadFunc, this, root, i);
    }

    //詰み探索を立ち上げ
    mate_searcher_.stop_signal = false;
    std::thread mate_thread([&](){ mate_searcher_.mateSearch(root, INT_MAX); });

    //終了を待つ
    for (std::thread& t : threads) {
        t.join();
    }

    mate_searcher_.stop_signal = true;
    mate_thread.join();

    //読み筋などの情報出力
    printUSIInfo();

    //行動選択
    if (root.turnNumber() < search_options_.random_turn) {
        std::vector<FloatType> distribution(curr_node.moves.size());
        if (search_options_.temperature_x1000 == 0) {
            //探索回数を正規化した分布に従って行動選択
            for (uint64_t i = 0; i < curr_node.moves.size(); i++) {
                distribution[i] = (FloatType) curr_node.N[i] / curr_node.sum_N;
                assert(0.0 <= distribution[i] && distribution[i] <= 1.0);
            }
        } else {
            //価値のソフトマックス分布に従って行動選択
            std::vector<FloatType> Q(curr_node.moves.size());
            for (uint64_t i = 0; i < curr_node.moves.size(); i++) {
                Q[i] = hash_table_.expQfromNext(curr_node, i);
            }
            distribution = softmax(Q, search_options_.temperature_x1000 / 1000.0f);
        }

        return curr_node.moves[randomChoose(distribution)];
    } else {
        //探索回数最大の手を選択
        int32_t best_index = std::max_element(curr_node.N.begin(), curr_node.N.end()) - curr_node.N.begin();
        return curr_node.moves[best_index];
    }
}

bool SearcherForPlay::shouldStop() {
    if (stop_signal) {
        return true;
    }

    //時間のチェック
    auto now_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now_time - start_);
    if (elapsed.count() >= time_limit_) {
        return true;
    }

    //ハッシュテーブルの容量チェック
    //並列化しているのでいくらか容量には余裕を持って確認しておかないといけない
    if (!hash_table_.hasEmptyEntries(search_options_.thread_num_per_gpu * search_options_.gpu_num)) {
        return true;
    }

    //探索回数のチェック
//    int32_t max1 = 0, max2 = 0;
//    for (int32_t i = 0; i < hash_table_[root_index_].moves.size(); i++) {
//        int32_t num = hash_table_[root_index_].N[i] + hash_table_[root_index_].virtual_N[i];
//        if (num > max1) {
//            max2 = max1;
//            max1 = num;
//        } else if (num > max2) {
//            max2 = num;
//        }
//    }
//    int32_t remainder = node_limit_ - (hash_table_[root_index_].sum_N + hash_table_[root_index_].virtual_sum_N);
//    return max1 - max2 >= remainder;

    int32_t search_num = hash_table_[hash_table_.root_index].sum_N + hash_table_[hash_table_.root_index].virtual_sum_N;
    return search_num >= node_limit_;
}

void SearcherForPlay::gpuThreadFunc(const Position& root, int64_t gpu_id) {
    //workerを立ち上げ
    std::vector<std::thread> threads(search_options_.thread_num_per_gpu);
    for (int64_t i = 0; i < search_options_.thread_num_per_gpu; i++) {
        threads[i] = std::thread(&SearcherForPlay::workerThreadFunc, this, root, gpu_id, i);
    }

    //終了を待つ
    for (std::thread& t : threads) {
        t.join();
    }
}

void SearcherForPlay::workerThreadFunc(Position root, int64_t gpu_id, int64_t thread_id) {
    //このスレッドに対するキューを参照で取る
    GPUQueue& gpu_queue = gpu_queues_[gpu_id][thread_id];

    //限界に達するまで探索を繰り返す
    while (!shouldStop()) {
        //キューをクリア
        gpu_queue.inputs.clear();
        gpu_queue.hash_tables.clear();
        gpu_queue.indices.clear();

        hash_table_[hash_table_.root_index].mutex.lock();
        auto now_time = std::chrono::steady_clock::now();
        auto elapsed_msec = std::chrono::duration_cast<std::chrono::milliseconds>(now_time - start_).count();
        if (elapsed_msec >= next_print_time_) {
            next_print_time_ += next_print_time_;
            printUSIInfo();
        }
        hash_table_[hash_table_.root_index].mutex.unlock();

        //評価要求を貯める
        for (int64_t i = 0; i < search_options_.search_batch_size && !shouldStop(); i++) {
            searchers_[gpu_id][thread_id].select(root);
        }

        if (shouldStop()) {
            break;
        }

        //評価要求をGPUで計算
        if (!gpu_queue.inputs.empty()) {
            torch::NoGradGuard no_grad_guard;
            gpu_mutexes_[gpu_id].lock();
            std::pair<std::vector<PolicyType>, std::vector<ValueType>> y = neural_networks_[gpu_id]->policyAndValueBatch(gpu_queue.inputs);
            gpu_mutexes_[gpu_id].unlock();

            //書き込み
            for (uint64_t i = 0; i < gpu_queue.indices.size(); i++) {
                std::unique_lock<std::mutex> lock(hash_table_[gpu_queue.indices[i]].mutex);
                HashEntry& curr_node = hash_table_[gpu_queue.indices[i]];
                curr_node.nn_policy.resize(curr_node.moves.size());
                for (uint64_t j = 0; j < curr_node.moves.size(); j++) {
                    curr_node.nn_policy[j] = y.first[i][curr_node.moves[j].toLabel()];
                }
                curr_node.nn_policy = softmax(curr_node.nn_policy);
                curr_node.value = y.second[i];
                curr_node.evaled = true;
            }
        }

        //バックアップ
        searchers_[gpu_id][thread_id].backupAll();
    }
}

std::vector<Move> SearcherForPlay::getPV() const {
    std::vector<Move> pv;
    for (Index index = hash_table_.root_index; index != HashTable::NOT_EXPANDED && !hash_table_[index].moves.empty();) {
        const std::vector<int32_t>& N = hash_table_[index].N;
        Index next_index = (int32_t)(std::max_element(N.begin(), N.end()) - N.begin());
        pv.push_back(hash_table_[index].moves[next_index]);
        index = hash_table_[index].child_indices[next_index];
    }

    return pv;
}

void SearcherForPlay::printUSIInfo() const {
    const HashEntry& curr_node = hash_table_[hash_table_.root_index];

    int32_t best_index = (std::max_element(curr_node.N.begin(), curr_node.N.end()) - curr_node.N.begin());

    //選択した着手の勝率の算出
    FloatType best_value = hash_table_.expQfromNext(curr_node, best_index);

#ifdef USE_CATEGORICAL
    //分布の表示
    constexpr int64_t gather_num = 3;
    for (int64_t i = 0; i < BIN_SIZE / gather_num; i++) {
        double p = 0.0;
        for (int64_t j = 0; j < gather_num; j++) {
            p += hash_table_.QfromNextValue(curr_node, best_index)[i * gather_num + j];
        }
        printf("info string [%+6.2f:%06.2f%%]:", MIN_SCORE + VALUE_WIDTH * (gather_num * i + 1.5), p * 100);
        for (int64_t j = 0; j < p * 50; j++) {
            printf("*");
        }
        printf("\n");
    }
#endif
    //探索にかかった時間を求める
    auto finish_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_);
    int64_t ela = elapsed.count();

    printf("info nps %d time %d nodes %d hashfull %d score cp %d pv ",
           (int32_t)(curr_node.sum_N * 1000LL / std::max(ela, (int64_t)1)),
           (int32_t)(ela),
           curr_node.sum_N,
           (int32_t) (hash_table_.getUsageRate() * 1000),
           (int32_t) (best_value * 1000));

    for (Move move : getPV()) {
        std::cout << move << " ";
    }
    std::cout << std::endl;

    if (search_options_.print_policy_num > 0) {
        //まず各指し手の価値を取得
        std::vector<FloatType> Q(curr_node.moves.size());
        for (uint64_t i = 0; i < curr_node.moves.size(); i++) {
            Q[i] = hash_table_.expQfromNext(curr_node, i);
        }
        std::vector<FloatType> softmaxed_Q = softmax(Q, search_options_.temperature_x1000 / 1000.f);

        //ソートするために構造体を準備
        struct MoveWithInfo {
            Move move;
            int32_t N;
            FloatType nn_output_policy, Q, softmaxed_Q;
#ifdef USE_CATEGORICAL
            FloatType prob_over_best_Q;
#endif
            bool operator<(const MoveWithInfo& rhs) const {
                return Q < rhs.Q;
            }
            bool operator>(const MoveWithInfo& rhs) const {
                return Q > rhs.Q;
            }
        };

        std::vector<MoveWithInfo> moves_with_info(curr_node.moves.size());
        for (uint64_t i = 0; i < curr_node.moves.size(); i++) {
            moves_with_info[i].move = curr_node.moves[i];
            moves_with_info[i].nn_output_policy = curr_node.nn_policy[i];
            moves_with_info[i].N = curr_node.N[i];
            moves_with_info[i].Q = Q[i];
            moves_with_info[i].softmaxed_Q = softmaxed_Q[i];
#ifdef USE_CATEGORICAL
            moves_with_info[i].prob_over_best_Q = 0;
            for (int32_t j = std::min(valueToIndex(best_value) + 1, BIN_SIZE - 1); j < BIN_SIZE; j++) {
                moves_with_info[i].prob_over_best_Q += hash_table_.QfromNextValue(curr_node, i)[j];
            }
#endif
        }
        std::sort(moves_with_info.begin(), moves_with_info.end());

        //指定された数だけ価値が高い順に表示する
        //GUIを通すと後に出力したものが上に来るので昇順ソートしたものを出力すれば上から降順になる
        for (uint64_t i = std::max((int64_t)0, (int64_t)curr_node.moves.size() - search_options_.print_policy_num);
                      i < curr_node.moves.size(); i++) {
#ifdef USE_CATEGORICAL
            printf("info string %03lu  %05.1f  %05.1f  %05.1f  %+0.3f  %05.1f ", curr_node.moves.size() - i,
                   moves_with_info[i].nn_output_policy * 100.0,
                   moves_with_info[i].N * 100.0 / curr_node.sum_N,
                   moves_with_info[i].softmaxed_Q * 100,
                   moves_with_info[i].Q,
                   moves_with_info[i].prob_over_best_Q * 100);
#else
            printf("info string %03lu  %05.1f  %05.1f  %05.1f  %+0.3f  ", curr_node.moves.size() - i,
                                                                          moves_with_info[i].nn_output_policy * 100.0,
                                                                          moves_with_info[i].N * 100.0 / curr_node.sum_N,
                                                                          moves_with_info[i].softmaxed_Q * 100,
                                                                          moves_with_info[i].Q);
#endif
            moves_with_info[i].move.print();
        }
#ifdef USE_CATEGORICAL
        std::cout << "info string 順位 NN出力 探索割合 価値分布 価値 最善超え確率" << std::endl;
#else
        std::cout << "info string 順位 NN出力 探索割合 価値分布 価値" << std::endl;
#endif
    }
}