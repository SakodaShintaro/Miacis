#include"parallel_MCTSearcher.hpp"
#include"searcher_common.hpp"

//-----------------------------
//    並列と直列に共通するもの
//-----------------------------
bool ParallelMCTSearcher::isTimeOver() {
    //時間のチェック
    auto now_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now_time - start_);
    return (elapsed.count() >= usi_option.limit_msec - usi_option.byoyomi_margin);
}

bool ParallelMCTSearcher::shouldStop() {
    return isTimeOver() || !hash_table_.hasEnoughSize();
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

//--------------------
//    並列化する実装
//--------------------
#ifdef USE_PARALLEL_SEARCHER

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

#else
//--------------------
//    直列化する実装
//--------------------
Move ParallelMCTSearcher::think(Position& root) {
    //思考開始時間をセット
    start_ = std::chrono::steady_clock::now();

    //古いハッシュを削除
    hash_table_.deleteOldHash(root, false);

    //キューの初期化
    for (int32_t i = 0; i < WORKER_NUM; i++) {
        input_queues_[i].clear();
        index_queues_[i].clear();
        route_queues_[i].clear();
        action_queues_[i].clear();
    }

    //ルートノードの展開:0番目のキューを使う
    std::stack<Index> dummy;
    std::stack<int32_t> dummy2;
    current_root_index_ = expandNode(root, dummy, dummy2, 0);
    auto& current_node = hash_table_[current_root_index_];

    //合法手が0だったら投了
    if (current_node.child_num == 0) {
        return NULL_MOVE;
    }

    //GPUで計算:child_num == 1のときはいらなそうだけど
    if (input_queues_[0].empty()) {
        //繰り返しなどでキューに送られなかった
        //ルートノードでは強制的に評価
        const auto feature = root.makeFeature();
        input_queues_[0].insert(input_queues_[0].end(), feature.begin(), feature.end());
    }
#ifdef USE_LIBTORCH
    auto y = evaluator_->policyAndValueBatch(input_queues_[0]);
#else
    auto y = evaluator_.policyAndValueBatch(input_queues_[0]);
#endif

    //ルートノードへ書き込み
    current_node.nn_rates.resize(static_cast<uint64_t>(current_node.child_num));
    for (int32_t i = 0; i < current_node.child_num; i++) {
        current_node.nn_rates[i] = y.first[0][current_node.legal_moves[i].toLabel()];
    }
    current_node.nn_rates = softmax(current_node.nn_rates);
    //valueは使わないはずだけど気分で
    current_node.value = y.second[0];

    //thread_numをWORKER_NUMの倍数にする
    thread_num_ = thread_num_ / WORKER_NUM * WORKER_NUM;
    if (thread_num_ == 0) {
        thread_num_ = WORKER_NUM;
    }

    //初期化
    playout_num_ = 0;

    //workerを立ち上げ
    std::vector<std::thread> threads(static_cast<uint64_t>(WORKER_NUM));
    for (int32_t i = 0; i < WORKER_NUM; i++) {
        threads[i] = std::thread(&ParallelMCTSearcher::parallelUctSearch, this, root, i);
    }

    //終了を待つ
    for (auto& t : threads) {
        t.join();
    }

    const auto& child_move_counts = current_node.child_move_counts;

    printUSIInfo();
    root.print(true);
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

void ParallelMCTSearcher::parallelUctSearch(Position root, int32_t id) {
    //限界に達するまで探索を繰り返す
    while (playout_num_ < usi_option.playout_limit && !shouldStop()) {
        //キューをクリア
        input_queues_[id].clear();
        index_queues_[id].clear();
        route_queues_[id].clear();
        action_queues_[id].clear();
        redundancy_num_[id].clear();

        //評価要求を貯める
        for (int32_t i = 0; i < thread_num_ / WORKER_NUM; i++) {
            //1回探索
            onePlay(root, id);

            if (playout_num_++ >= usi_option.playout_limit || shouldStop()) {
                break;
            }
        }

        //評価要求をGPUで計算
        if (!index_queues_[id].empty()) {
            lock_expand_.lock();
#ifdef USE_LIBTORCH
            auto y = evaluator_->policyAndValueBatch(input_queues_[id]);
#else
            auto y = evaluator_.policyAndValueBatch(input_queues_[id]);
#endif
            lock_expand_.unlock();

            //書き込み
            for (int32_t i = 0; i < index_queues_[id].size(); i++) {
                std::unique_lock<std::mutex> lock(lock_node_[index_queues_[id][i]]);
                auto& curr_node = hash_table_[index_queues_[id][i]];
                curr_node.nn_rates.resize(static_cast<uint64_t>(curr_node.child_num));
                for (int32_t j = 0; j < curr_node.legal_moves.size(); j++) {
                    curr_node.nn_rates[j] = y.first[i][curr_node.legal_moves[j].toLabel()];
                }
                curr_node.nn_rates = softmax(curr_node.nn_rates);
                curr_node.value = y.second[i];
                curr_node.evaled = true;
            }
        }

        //バックアップ
        for (int32_t i = 0; i < route_queues_[id].size(); i++) {
            backup(route_queues_[id][i], action_queues_[id][i], 1 - VIRTUAL_LOSS * redundancy_num_[id][i]);
            playout_num_ += 1 - redundancy_num_[id][i];
        }
        assert(hash_table_[current_root_index_].move_count = std::accumulate(hash_table_[current_root_index_].child_move_counts.begin(),
                hash_table_[current_root_index_].child_move_counts.end(), 0));
    }
}

void ParallelMCTSearcher::onePlay(Position &pos, int32_t id) {
    std::stack<Index> curr_indices;
    std::stack<int32_t> curr_actions;

    auto index = current_root_index_;
    //ルートでは合法手が一つはあるはず
    assert(hash_table_[index].child_num != 0);

    //未展開の局面に至るまで遷移を繰り返す
    while(index != UctHashTable::NOT_EXPANDED) {
        std::unique_lock<std::mutex> lock(lock_node_[index]);

        if (hash_table_[index].child_num == 0) {
            //詰みの場合抜ける
            break;
        }

        Score repeat_score;
        if (index != current_root_index_ && pos.isRepeating(repeat_score)) {
            //繰り返しが発生している場合も抜ける
            break;
        }

        if (hash_table_[index].nn_rates.size() != hash_table_[index].legal_moves.size()) {
            //policyが展開されていなかったら抜ける
            break;
        }

        //状態を記録
        curr_indices.push(index);

        //選択
        auto action = selectMaxUcbChild(hash_table_[index]);

        //取った行動を記録
        curr_actions.push(action);

        //VIRTUAL_LOSSの追加
        hash_table_[index].move_count += VIRTUAL_LOSS;
        hash_table_[index].child_move_counts[action] += VIRTUAL_LOSS;

        //遷移
        pos.doMove(hash_table_[index].legal_moves[action]);

        //index更新
        index = hash_table_[index].child_indices[action];
    }

    //expandNode内でこれらの情報は壊れる可能性があるので保存しておく
    index = curr_indices.top();
    auto action = curr_actions.top();
    auto move_num = curr_actions.size();

    //今の局面を展開・GPUに評価依頼を投げる
    auto leaf_index = expandNode(pos, curr_indices, curr_actions, id);

    //葉の直前ノードを更新
    lock_node_[index].lock();
    hash_table_[index].child_indices[action] = leaf_index;
    lock_node_[index].unlock();

    //バックアップはGPU計算後にやるので局面だけ戻す
    for (int32_t i = 0; i < move_num; i++) {
        pos.undo();
    }
}

Index ParallelMCTSearcher::expandNode(Position& pos, std::stack<int32_t>& indices, std::stack<int32_t>& actions, int32_t id) {
    //全体をロック.このノードだけではなく探す部分も含めてなので
    std::unique_lock<std::mutex> lock(lock_expand_);

    auto index = hash_table_.findSameHashIndex(pos.hash_value(), static_cast<int16_t>(pos.turn_number()));

    // 合流先が検知できればそれを返す
    if (index != hash_table_.size()) {
        indices.push(index);
        if (hash_table_[index].evaled) {
            //評価済みならば,前回までのループでここへ違う経路で到達していたか,終端状態であるかのいずれか
            //どちらの場合でもバックアップして良い,と思う
            //GPUに送らないのでこのタイミングでバックアップを行う
            backup(indices, actions, 1 - VIRTUAL_LOSS);
        } else {
            //評価済みではないけどここへ到達したならば,同じループの中で到達があったということ
            //全く同じ経路のものがあるかどうか確認
            auto itr = std::find(route_queues_[id].begin(), route_queues_[id].end(), indices);
            if (itr == route_queues_[id].end()) {
                //同じものはなかった
                route_queues_[id].push_back(indices);
                action_queues_[id].push_back(actions);
                redundancy_num_[id].push_back(1);
            } else {
                //同じものがあった
                redundancy_num_[id][itr - route_queues_[id].begin()]++;
            }
        }
        return index;
    }

    // 空のインデックスを探す
    index = hash_table_.searchEmptyIndex(pos.hash_value(), static_cast<int16_t>(pos.turn_number()));

    //経路として記録
    indices.push(index);

    auto& current_node = hash_table_[index];

    // 候補手の展開
    current_node.legal_moves = pos.generateAllMoves();
    current_node.child_num = (uint32_t)current_node.legal_moves.size();
    current_node.child_indices.assign(current_node.child_num, UctHashTable::NOT_EXPANDED);
    current_node.child_move_counts.assign(current_node.child_num, 0);

    // 現在のノードの初期化
    current_node.move_count = 0;
    current_node.evaled = false;
#ifdef USE_CATEGORICAL
    //TODO:正しく初期化できているか確認すること
    current_node.child_wins.assign(static_cast<unsigned long>(current_node.child_num), std::array<float, BIN_SIZE>{});
    current_node.value = std::array<float, BIN_SIZE>{};
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        //current_node.value[i] = 0.0;
        std::cout << current_node.value[i] << std::endl;
        for (int32_t j = 0; j < current_node.child_num; j++) {
            current_node.child_wins[j][i] = 0.0;
        }
    }
#else
    current_node.value = 0.0;
    current_node.child_wins.assign(static_cast<unsigned long>(current_node.child_num), 0.0);
#endif

    // ノードを評価
    Score repeat_score;
    if (pos.isRepeating(repeat_score)) {
        //繰り返し
#ifdef USE_CATEGORICAL
        current_node.value = onehotDist(repeat_score);
#else
        current_node.value = repeat_score;
#endif
        current_node.evaled = true;
        //GPUに送らないのでこのタイミングでバックアップを行う
        backup(indices, actions, 1 - VIRTUAL_LOSS);
    } else if (current_node.child_num > 0) {
        //GPUへ計算要求を投げる
        //特徴量の追加
        auto this_feature = pos.makeFeature();
        input_queues_[id].insert(input_queues_[id].end(), this_feature.begin(), this_feature.end());
        index_queues_[id].push_back(index);
        //バックアップ要求も投げる
        route_queues_[id].push_back(indices);
        action_queues_[id].push_back(actions);
        redundancy_num_[id].push_back(1);
    } else {
        if (pos.lastMove().isDrop() && (kind(pos.lastMove().subject()) == PAWN)) {
            //打ち歩詰めなので勝ち
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
        //GPUに送らないのでこのタイミングでバックアップを行う
        backup(indices, actions, 1 - VIRTUAL_LOSS);
    }

    return index;
}

void ParallelMCTSearcher::backup(std::stack<int32_t>& indices, std::stack<int32_t>& actions, int32_t add_num) {
    assert(indices.size() == actions.size() + 1);
    auto leaf = indices.top();
    indices.pop();
    auto value = hash_table_[leaf].value;

    //バックアップ
    while (!actions.empty()) {
        auto index = indices.top();
        indices.pop();

        auto action = actions.top();
        actions.pop();

        //手番が変わっているので反転
        value = reverse(value);

        // 探索結果の反映
        lock_node_[index].lock();
        hash_table_[index].child_wins[action] += value;
        hash_table_[index].move_count += add_num;
        hash_table_[index].child_move_counts[action] += add_num;
        assert(hash_table_[index].child_num != 0);
        lock_node_[index].unlock();
    }
}

#endif