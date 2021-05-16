#include "searcher_for_play.hpp"
#include <thread>

struct MoveWithScore {
public:
    Move move;
    float score;
    bool operator<(const MoveWithScore& rhs) const { return score < rhs.score; }
    bool operator>(const MoveWithScore& rhs) const { return score > rhs.score; }
};

SearcherForPlay::SearcherForPlay(const SearchOptions& search_options)
    : stop_signal(false), search_options_(search_options),
      hash_table_(search_options.USI_Hash * 1024 * 1024 / (120 * search_options.hold_moves_num)),
      mate_searcher_(hash_table_, search_options) {
    //GPUを準備
    for (int64_t i = 0; i < search_options.gpu_num; i++) {
        neural_networks_.emplace_back();
        neural_networks_[i].load(i, search_options);
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

    if (search_options.output_log_file) {
        log_file_.open("search_log.txt");
    }

#ifdef SHOGI
    book_.open(search_options.book_file_name);
#endif
}

Move SearcherForPlay::think(Position& root, int64_t time_limit) {
    //思考開始時間をセット
    start_ = std::chrono::steady_clock::now();

#ifdef SHOGI
    float score{};
    if (!root.isRepeating(score) && search_options_.use_book && book_.hasEntry(root)) {
        Move move = book_.pickOne(root, search_options_.book_temperature_x1000);
        return root.transformValidMove(move);
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

    if (search_options_.output_log_file) {
        log_file_ << "startSearch" << std::endl;
        log_file_ << root.toStr() << std::endl;
    }

    //GPUで計算
    if (curr_node.nn_policy.size() != curr_node.moves.size()) {
        if (gpu_queues_[0][0].inputs.empty()) {
            std::vector<float> feature = root.makeFeature();
            gpu_queues_[0][0].inputs.insert(gpu_queues_[0][0].inputs.begin(), feature.begin(), feature.end());
        }
        torch::NoGradGuard no_grad_guard;
        std::pair<std::vector<PolicyType>, std::vector<ValueType>> y =
            neural_networks_[0].policyAndValueBatch(gpu_queues_[0][0].inputs);

        //ルートノードへ書き込み
        curr_node.nn_policy.resize(curr_node.moves.size());
        for (uint64_t i = 0; i < curr_node.moves.size(); i++) {
            curr_node.nn_policy[i] = y.first[0][curr_node.moves[i].toLabel()];
        }
        curr_node.nn_policy = softmax(curr_node.nn_policy, search_options_.policy_temperature_x1000 / 1000.0f);
        curr_node.value = y.second[0];
        curr_node.evaled = true;
    }

    if (search_options_.output_log_file) {
        outputInfo(log_file_, 1);
    }

    //GPUに付随するスレッドを立ち上げ
    std::vector<std::thread> threads(search_options_.gpu_num);
    for (int64_t i = 0; i < search_options_.gpu_num; i++) {
        threads[i] = std::thread(&SearcherForPlay::gpuThreadFunc, this, root, i);
    }

    //詰み探索を立ち上げ
    mate_searcher_.stop_signal = false;
    std::thread mate_thread([&]() { mate_searcher_.mateSearch(root, INT_MAX); });

    //終了を待つ
    for (std::thread& t : threads) {
        t.join();
    }

    mate_searcher_.stop_signal = true;
    mate_thread.join();

    //読み筋などの情報出力
    if (search_options_.print_info) {
        outputInfo(std::cout, 3);
    }
    if (search_options_.output_log_file) {
        outputInfo(log_file_, 1);
        log_file_ << "endSearch" << std::endl;
    }

    //行動選択
    if (root.turnNumber() <= search_options_.random_turn) {
        std::vector<float> distribution(curr_node.moves.size());
        if (search_options_.temperature_x1000 == 0) {
            //探索回数を正規化した分布に従って行動選択
            for (uint64_t i = 0; i < curr_node.moves.size(); i++) {
                distribution[i] = (float)curr_node.N[i] / curr_node.sum_N;
                assert(0.0 <= distribution[i] && distribution[i] <= 1.0);
            }
        } else {
            //価値のソフトマックス分布に従って行動選択
            std::vector<float> Q(curr_node.moves.size());
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
    if (!hash_table_.hasEmptyEntries(search_options_.search_batch_size * search_options_.thread_num_per_gpu *
                                     search_options_.gpu_num)) {
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
            if (search_options_.print_info) {
                outputInfo(std::cout, 3);
            }
            if (search_options_.output_log_file) {
                outputInfo(log_file_, 1);
            }
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
            std::tuple<torch::Tensor, torch::Tensor> output = neural_networks_[gpu_id].infer(gpu_queue.inputs);
            gpu_mutexes_[gpu_id].unlock();
            std::pair<std::vector<PolicyType>, std::vector<ValueType>> y = tensorToVector(output);

            //書き込み
            for (uint64_t i = 0; i < gpu_queue.indices.size(); i++) {
                std::unique_lock<std::mutex> lock(hash_table_[gpu_queue.indices[i]].mutex);
                HashEntry& curr_node = hash_table_[gpu_queue.indices[i]];

                //Policyの大きい順にsearch_option.hold_moves_num個だけ残す
                std::vector<MoveWithScore> moves_with_score(curr_node.moves.size());
                for (uint64_t j = 0; j < curr_node.moves.size(); j++) {
                    moves_with_score[j].move = curr_node.moves[j];
                    moves_with_score[j].score = y.first[i][curr_node.moves[j].toLabel()];
                }

                std::sort(moves_with_score.begin(), moves_with_score.end(), std::greater<>());

                uint64_t moves_num = std::min((int64_t)curr_node.moves.size(), search_options_.hold_moves_num);
                curr_node.moves.resize(moves_num);
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
                    curr_node.moves[j] = moves_with_score[j].move;
                    curr_node.nn_policy[j] = moves_with_score[j].score;
                }
                curr_node.nn_policy = softmax(curr_node.nn_policy, search_options_.policy_temperature_x1000 / 1000.0f);
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

void SearcherForPlay::outputInfo(std::ostream& ost, int64_t gather_num) const {
    //表示の設定
    ost << std::fixed << std::setfill('0');

    //ノードの取得
    const HashEntry& curr_node = hash_table_[hash_table_.root_index];

    //最善手（探索回数最大手）の価値を計算
    int32_t best_index = (std::max_element(curr_node.N.begin(), curr_node.N.end()) - curr_node.N.begin());
    float best_value = hash_table_.expQfromNext(curr_node, best_index);

#ifdef USE_CATEGORICAL
    //分布の表示
    //51分割を51行で表示すると見づらいのでいくらか領域をまとめる
    for (int64_t i = 0; i < BIN_SIZE / gather_num; i++) {
        float p = 0.0;

        //gather_num分だけ合算する
        for (int64_t j = 0; j < gather_num; j++) {
            p += curr_node.value[i * gather_num + j];
        }

        //表示
        ost << "info string [" << std::setprecision(2) << std::showpos << MIN_SCORE + VALUE_WIDTH * (gather_num * (i + 0.5))
            << ":" << std::setw(5) << std::setprecision(1) << std::noshowpos << p * 100 << "]:";
        for (int64_t j = 0; j < p * 50; j++) {
            ost << "*";
        }
        ost << std::endl;
    }
#endif

    //探索にかかった時間を求める
    auto finish_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_);
    int64_t ela = elapsed.count();

    //PVを取得
    std::vector<Move> pv = getPV();

    //表示
    // clang-format off
    ost << "info nps " << (int32_t)(curr_node.sum_N * 1000LL / std::max(ela, (int64_t)1))
        << " time " << ela
        << " nodes " << curr_node.sum_N
        << " depth " << pv.size()
        << " hashfull " << (int32_t)(hash_table_.getUsageRate() * 1000)
        << " score cp " << (int32_t)(best_value * 5000)
        << " pv ";
    // clang-format on
    for (Move move : pv) {
        ost << move << " ";
    }
    ost << std::endl;

    //指し手について表示する必要がないならここで終了
    if (search_options_.print_policy_num <= 0) {
        return;
    }

    //まず各指し手の価値を取得
    std::vector<float> Q(curr_node.moves.size());
    for (uint64_t i = 0; i < curr_node.moves.size(); i++) {
        Q[i] = hash_table_.expQfromNext(curr_node, i);
    }
    std::vector<float> softmaxed_Q = softmax(Q, std::max(search_options_.temperature_x1000, (int64_t)1) / 1000.f);

    //ソートするために構造体を準備
    struct MoveWithInfo {
        Move move;
        int32_t N;
        float nn_output_policy, Q, softmaxed_Q;
#ifdef USE_CATEGORICAL
        float prob_over_best_Q;
#endif
        bool operator<(const MoveWithInfo& rhs) const { return Q < rhs.Q; }
        bool operator>(const MoveWithInfo& rhs) const { return Q > rhs.Q; }
    };

    //全指し手について情報を集めて価値順にソート
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
        // clang-format off
        ost << "info string " << std::setw(3) << curr_node.moves.size() - i << "  "
                              << std::setw(5) << std::setprecision(1) << moves_with_info[i].nn_output_policy * 100.0 << "  "
                              << std::setw(5) << std::setprecision(1) << moves_with_info[i].N * 100.0 / curr_node.sum_N << "  "
                              << std::setw(5) << std::setprecision(1) << moves_with_info[i].softmaxed_Q * 100.0 << "  "
                              << std::setw(5) << std::setprecision(3) << std::showpos << moves_with_info[i].Q << std::noshowpos << "  "
#ifdef USE_CATEGORICAL
                              << std::setw(5) << std::setprecision(1) << moves_with_info[i].prob_over_best_Q * 100.0 << "  "
#endif
                              << moves_with_info[i].move.toPrettyStr() << std::endl;
        // clang-format on
    }
#ifdef USE_CATEGORICAL
    ost << "info string 順位 NN出力 探索割合 価値分布 価値 最善超え確率" << std::endl;
#else
    ost << "info string 順位 NN出力 探索割合 価値分布 価値" << std::endl;
#endif

    //設定をデフォルトに戻す
    ost.clear();
}