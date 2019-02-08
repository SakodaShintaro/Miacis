#include"parallel_MCTSearcher.hpp"
#include"neural_network.hpp"

Move ParallelMCTSearcher::think(Position& root) {
    //思考開始時間をセット
    start_ = std::chrono::steady_clock::now();

    //古いハッシュを削除
    hash_table_.deleteOldHash(root, false);

    clearEvalQueue();

    //ルートノードの展開
    current_root_index_ = expandNode(root);
    auto& current_node = hash_table_[current_root_index_];

    //合法手が0だったら投了
    if (current_node.child_num == 0) {
        return NULL_MOVE;
    }

    //初期化
    playout_num_ = 0;

    running_ = true;

    std::thread calc_nn(&ParallelMCTSearcher::evalNode, this);

    while (!hash_table_[current_root_index_].evaled) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }

    std::vector<std::thread> threads(static_cast<unsigned long>(thread_num_));
    for (auto& t : threads) {
        t = std::thread(&ParallelMCTSearcher::parallelUctSearch, this, root);
    }

    for (auto& t : threads) {
        t.join();
    }

    running_ = false;

    calc_nn.join();

    const auto& child_move_counts = current_node.child_move_counts;

    printUSIInfo();
    root.print(false);
    for (int32_t i = 0; i < current_node.child_num; i++) {
        double nn = 100.0 * current_node.nn_rates[i];
        double p  = 100.0 * child_move_counts[i] / current_node.move_count;
#ifdef USE_CATEGORICAL
        double v = (child_move_counts[i] > 0 ? expOfValueDist(current_node.child_wins[i]) / child_move_counts[i] : MIN_SCORE);
#else
        double v = (child_move_counts[i] > 0 ? current_node.child_wins[i] / child_move_counts[i] : MIN_SCORE);
#endif
        printf("%3d  %4.1f  %4.1f  %+.3f  ", i, nn, p, v);
        current_node.legal_moves[i].print();
    }

    //探索回数最大の手を選択する
    int32_t best_index = (int32_t)(std::max_element(child_move_counts.begin(), child_move_counts.end()) - child_move_counts.begin());

    //選択した手の探索回数は少なくとも1以上であることを前提とする
    assert(child_move_counts[best_index] != 0);

    //選択した着手の勝率の算出
#ifdef USE_CATEGORICAL
    double best_wp = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        double v = current_node.child_wins[best_index][i] / child_move_counts[best_index];
        best_wp += VALUE_WIDTH * (0.5 + i) * v;
    }
#else
    auto best_wp = current_node.child_wins[best_index] / child_move_counts[best_index];
#endif

    //訪問回数に基づいた分布を得る
    std::vector<CalcType> distribution(static_cast<unsigned long>(current_node.child_num));
    for (int32_t i = 0; i < current_node.child_num; i++) {
        distribution[i] = (CalcType)child_move_counts[i] / current_node.move_count;
        assert(0.0 <= distribution[i] && distribution[i] <= 1.0);
    }

    //最善手
    Move best_move = (root.turn_number() < usi_option.random_turn ?
                      current_node.legal_moves[randomChoose(distribution)] :
                      current_node.legal_moves[best_index]);

    best_move.score = (Score)(best_wp);

    return best_move;
}

ValueType ParallelMCTSearcher::uctSearch(Position & pos, Index current_index) {
    auto& current_node = hash_table_[current_index];

    if (current_node.child_num == 0) {
#ifdef USE_CATEGORICAL
        std::array<CalcType, BIN_SIZE> lose_value_dist;
        for (int32_t i = 0; i < BIN_SIZE; i++) {
            lose_value_dist[i] = (i == 0 ? 1.0f : 0.0f);
        }
        return lose_value_dist;
#else
        return MIN_SCORE;
#endif
    }

    auto& child_indices = current_node.child_indices;

    while (!current_node.evaled) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }

    lock_node_[current_index].lock();

    // UCB値が最大の手を求める
    auto next_index = selectMaxUcbChild(current_node);

    current_node.move_count += VIRTUAL_LOSS;
    current_node.child_move_counts[next_index] += VIRTUAL_LOSS;

    // 選んだ手を着手
    pos.doMove(current_node.legal_moves[next_index]);

    ValueType result;
    Score score;
    // ノードの展開の確認
    if (pos.isRepeating(score)) {
        lock_node_[current_index].unlock();

#ifdef USE_CATEGORICAL
        result = onehotDist(score);
#else
        result = score;
#endif
    } else if (child_indices[next_index] == UctHashTable::NOT_EXPANDED) {
        // ノードの展開:ロック
        lock_expand_.lock();
        auto index = expandNode(pos);
        lock_expand_.unlock();

        lock_node_[current_index].unlock();

        child_indices[next_index] = index;
        while (!hash_table_[index].evaled) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }

        result = hash_table_[index].value;
    } else {
        //ロック解除
        lock_node_[current_index].unlock();

        // 手番を入れ替えて1手深く読む
        result = uctSearch(pos, child_indices[next_index]);
    }
    //手番が変わっているので反転
    result = reverse(result);

    // 探索結果の反映
    lock_node_[current_index].lock();
    current_node.move_count += 1 - VIRTUAL_LOSS;
    current_node.child_wins[next_index] += result;
    current_node.child_move_counts[next_index] += 1 - VIRTUAL_LOSS;
    lock_node_[current_index].unlock();

    // 手を戻す
    pos.undo();

    return result;
}

void ParallelMCTSearcher::parallelUctSearch(Position root) {
    //プレイアウトを繰り返す
    //探索回数が閾値を超える、または探索が打ち切られたらループを抜ける
    while (playout_num_ < usi_option.playout_limit) {
        //探索回数を1回増やす
        playout_num_++;

        //1回プレイアウト
        uctSearch(root, current_root_index_);

        //onePlay(root);

        //探索を打ち切るか確認
        if (shouldStop() || !hash_table_.hasEnoughSize()) {
            break;
        }
    }
}

Index ParallelMCTSearcher::expandNode(Position& pos) {
    auto index = hash_table_.findSameHashIndex(pos.hash_value(), static_cast<int16_t>(pos.turn_number()));

    // 合流先が検知できればそれを返す
    if (index != hash_table_.size()) {
        return index;
    }

    // 空のインデックスを探す
    index = hash_table_.searchEmptyIndex(pos.hash_value(), static_cast<int16_t>(pos.turn_number()));

    auto& current_node = hash_table_[index];

    // 候補手の展開
    current_node.legal_moves = pos.generateAllMoves();
    current_node.child_num = (uint32_t)current_node.legal_moves.size();
    current_node.child_indices = std::vector<int32_t>(current_node.child_num, UctHashTable::NOT_EXPANDED);
    current_node.child_move_counts = std::vector<int32_t>(current_node.child_num, 0);

    // 現在のノードの初期化
    current_node.move_count = 0;
    current_node.evaled = false;
#ifdef USE_CATEGORICAL
    current_node.child_wins = std::vector<std::array<CalcType, BIN_SIZE>>(current_node.child_num);
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        current_node.value[i] = 0.0;
        for (int32_t j = 0; j < current_node.child_num; j++) {
            current_node.child_wins[j][i] = 0.0;
        }
    }
#else
    current_node.value = 0.0;
    current_node.child_wins = std::vector<float>(static_cast<unsigned long>(current_node.child_num), 0.0);
#endif

    // ノードを評価
    if (current_node.child_num > 0) {
        auto this_feature = pos.makeFeature();
        current_features_.resize(current_features_.size() + this_feature.size());
        std::copy(this_feature.begin(), this_feature.end(), current_features_.end() - this_feature.size());
        current_hash_index_queue_.push_back(index);
    } else {
        if (pos.lastMove().isDrop() && (kind(pos.lastMove().subject()) == PAWN)) {
            //打ち歩詰め
#ifdef USE_CATEGORICAL
            for (int32_t i = 0; i < BIN_SIZE; i++) {
                current_node.value[i] = (i == BIN_SIZE - 1 ? 1.0f : 0.0f);
            }
#else
            current_node.value = MAX_SCORE;
#endif
        } else {
            //詰み
#ifdef USE_CATEGORICAL
            for (int32_t i = 0; i < BIN_SIZE; i++) {
                current_node.value[i] = (i == 0 ? 1.0f : 0.0f);
            }
#else
            current_node.value = MIN_SCORE;
#endif
        }
        current_node.evaled = true;
    }

    return index;
}

void ParallelMCTSearcher::evalNode() {
    bool enough_batch_size = true;

    while (running_) {
        lock_expand_.lock();
        if (current_features_.empty()) {
            lock_expand_.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            continue;
        }

        if (!enough_batch_size && current_features_.size() < thread_num_ / 2) {
            lock_expand_.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            enough_batch_size = true;
            continue;
        }

        enough_batch_size = false;

        //現在のキューを保存
        auto eval_features = current_features_;
        auto eval_hash_index_queue = current_hash_index_queue_;

        //カレントキューを入れ替える
        current_queue_index_ ^= 1;
        current_features_ = features_[current_queue_index_];
        current_features_.clear();
        current_hash_index_queue_ = hash_index_queues_[current_queue_index_];
        current_hash_index_queue_.clear();
        lock_expand_.unlock();

#ifdef USE_LIBTORCH
        auto result = evaluator_->policyAndValueBatch(eval_features);
#else
        auto result = evaluator_.policyAndValueBatch(eval_features);
#endif
        auto policies = result.first;
        auto values = result.second;

        for (int32_t i = 0; i < eval_hash_index_queue.size(); i++) {
            std::unique_lock<std::mutex> lock2(lock_node_[eval_hash_index_queue[i]]);

            //policyを設定
            auto& current_node = hash_table_[eval_hash_index_queue[i]];
            std::vector<float> legal_moves_policy(static_cast<unsigned long>(current_node.child_num));
            for (int32_t j = 0; j < current_node.child_num; j++) {
                legal_moves_policy[j] = policies[i][current_node.legal_moves[j].toLabel()];
            }
            current_node.nn_rates = softmax(legal_moves_policy);

            //valueを設定
            current_node.value = values[i];
            current_node.evaled = true;
        }
    }
}

bool ParallelMCTSearcher::isTimeOver() {
    //時間のチェック
    auto now_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now_time - start_);
    return (elapsed.count() >= usi_option.limit_msec - usi_option.byoyomi_margin);
}

bool ParallelMCTSearcher::shouldStop() {
    return isTimeOver();
//    if (isTimeOver()) {
//        return true;
//    }
//    return false;
//
//    // 探索回数が最も多い手と次に多い手を求める
//    int32_t max1 = 0, max2 = 0;
//    for (auto e : hash_table_[current_root_index_].child_move_counts) {
//        if (e > max1) {
//            max2 = max1;
//            max1 = e;
//        } else if (e > max2) {
//            max2 = e;
//        }
//    }
//
//    // 残りの探索を全て次善手に費やしても最善手を超えられない場合は探索を打ち切る
//    return (max1 - max2) > (usi_option.playout_limit - playout_num_);
}

std::vector<Move> ParallelMCTSearcher::getPV() const {
    std::vector<Move> pv;
    for (Index curr_node_index = current_root_index_; curr_node_index != UctHashTable::NOT_EXPANDED && hash_table_[curr_node_index].child_num != 0; ) {
        const auto& child_move_counts = hash_table_[curr_node_index].child_move_counts;
        Index next_index = (int32_t)(std::max_element(child_move_counts.begin(), child_move_counts.end()) - child_move_counts.begin());
        pv.push_back(hash_table_[curr_node_index].legal_moves[next_index]);
        curr_node_index = hash_table_[curr_node_index].child_indices[next_index];
    }

    return pv;
}

void ParallelMCTSearcher::printUSIInfo() const {
    const auto& current_node = hash_table_[current_root_index_];

    //探索にかかった時間を求める
    auto finish_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_);

    const auto& child_move_counts = current_node.child_move_counts;
    auto selected_index = (int32_t)(std::max_element(child_move_counts.begin(), child_move_counts.end()) - child_move_counts.begin());

    //選択した着手の勝率の算出
#ifdef USE_CATEGORICAL
    double best_wp = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        double v = current_node.child_wins[selected_index][i] / child_move_counts[selected_index];
        best_wp += VALUE_WIDTH * (0.5 + i) * v;
    }
#else
    auto best_wp = (child_move_counts[selected_index] == 0 ? 0.0
                                                           : current_node.child_wins[selected_index] / child_move_counts[selected_index]);
#endif

    //勝率を評価値に変換
    //int32_t cp = inv_sigmoid(best_wp, CP_GAIN);
    int32_t cp = (int32_t)(best_wp * 1000);

    printf("info nps %d time %d nodes %d hashfull %d score cp %d pv ",
           (int)(current_node.move_count * 1000 / std::max((long long)elapsed.count(), 1LL)),
           (int)(elapsed.count()),
           current_node.move_count,
           (int)(hash_table_.getUsageRate() * 1000),
           cp);

    auto pv = getPV();
    for (auto m : pv) {
        std::cout << m << " ";
    }
    std::cout << std::endl;
}

int32_t ParallelMCTSearcher::selectMaxUcbChild(const UctHashEntry & current_node) {
    const auto& child_move_counts = current_node.child_move_counts;

#ifdef USE_CATEGORICAL
    int32_t selected_index = (int32_t)(std::max_element(child_move_counts.begin(), child_move_counts.end()) - child_move_counts.begin());
    double best_wp = expOfValueDist(current_node.child_wins[selected_index]) / child_move_counts[selected_index];
#endif

    // ucb = Q(s, a) + U(s, a)
    // Q(s, a) = W(s, a) / N(s, a)
    // U(s, a) = C_PUCT * P(s, a) * sqrt(sum_b(B(s, b)) / (1 + N(s, a))
    constexpr double C_PUCT = 1.0;

    int32_t max_index = -1;
    double max_value = MIN_SCORE - 1;
    for (int32_t i = 0; i < current_node.child_num; i++) {
#ifdef USE_CATEGORICAL
        double Q;
        if (child_move_counts[i] == 0) {
            //中間を初期値とする
            Q = (MAX_SCORE + MIN_SCORE) / 2;
        } else {
            Q = 0.0;
            for (int32_t j = std::min((int32_t)(best_wp * BIN_SIZE) + 1, BIN_SIZE - 1); j < BIN_SIZE; j++) {
                Q += current_node.child_wins[i][j] / child_move_counts[i];
            }
        }
#else
        double Q = (child_move_counts[i] == 0 ? (MAX_SCORE + MIN_SCORE) / 2 : current_node.child_wins[i] / child_move_counts[i]);
#endif
        double U = std::sqrt(current_node.move_count + 1) / (child_move_counts[i] + 1);
        double ucb = Q + C_PUCT * current_node.nn_rates[i] * U;

        if (ucb > max_value) {
            max_value = ucb;
            max_index = i;
        }
    }
    assert(0 <= max_index && max_index < (int32_t)current_node.child_num);
    return max_index;
}

void ParallelMCTSearcher::onePlay(Position &pos) {
    std::stack<Index> indices;
    std::stack<int32_t> actions;

    auto index = current_root_index_;

    //未展開の局面に至るまで遷移を繰り返す
    while(index != UctHashTable::NOT_EXPANDED) {
        //状態を記録
        indices.push(index);

        //選択
        auto action = selectMaxUcbChild(hash_table_[index]);

        //取った行動を記録
        actions.push(action);

        //遷移
        pos.doMove(hash_table_[index].legal_moves[action]);

        //index更新
        index = hash_table_[index].child_indices[action];
    }

    //今の局面を展開・評価
    index = expandNode(pos);
    auto result = hash_table_[index].value;
    hash_table_[indices.top()].child_indices[actions.top()] = index;

    //バックアップ
    while (!actions.empty()) {
        pos.undo();
        index = indices.top();
        indices.pop();

        auto action = actions.top();
        actions.pop();

        //手番が変わっているので反転
        result = reverse(result);

        // 探索結果の反映
        hash_table_[index].child_wins[action] += result;
        hash_table_[index].move_count++;
        hash_table_[index].child_move_counts[action]++;
    }
}

void ParallelMCTSearcher::clearEvalQueue() {
    current_queue_index_ = 0;
    for (int32_t i = 0; i < 2; i++) {
        features_[i].clear();
        hash_index_queues_[i].clear();
    }
    current_features_ = features_[current_queue_index_];
    current_hash_index_queue_ = hash_index_queues_[current_queue_index_];
}