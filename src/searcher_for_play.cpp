#include "searcher_for_play.hpp"
#include "usi_options.hpp"
#include <thread>

Move SearcherForPlay::think(Position& root) {
    //思考開始時間をセット
    start_ = std::chrono::steady_clock::now();

    //古いハッシュを削除
    hash_table_.deleteOldHash(root, true);

    //次に表示するノード数を初期化
    next_print_node_num_ = usi_option.print_interval;

    //キューの初期化
    for (int32_t i = 0; i < thread_num_; i++) {
        input_queues_[i].clear();
        index_queues_[i].clear();
        route_queues_[i].clear();
        action_queues_[i].clear();
    }

    //ルートノードの展開:0番目のキューを使う
    std::stack<Index> dummy;
    std::stack<int32_t> dummy2;
    current_root_index_ = expand(root, dummy, dummy2, 0);
    auto& curr_node = hash_table_[current_root_index_];

    //合法手が0だったら投了
    if (curr_node.moves.empty()) {
        return NULL_MOVE;
    }

    //GPUで計算:moves.size() == 1のときはいらなそうだけど
    if (input_queues_[0].empty()) {
        //繰り返しなどでキューに送られなかった場合に空になることがありうる
        //ルートノードでは強制的に評価する
        const auto feature = root.makeFeature();
        input_queues_[0].insert(input_queues_[0].end(), feature.begin(), feature.end());
    }
    auto y = evaluator_->policyAndValueBatch(input_queues_[0]);

    //ルートノードへ書き込み
    curr_node.nn_policy.resize(curr_node.moves.size());
    for (int32_t i = 0; i < curr_node.moves.size(); i++) {
        curr_node.nn_policy[i] = y.first[0][curr_node.moves[i].toLabel()];
    }
    curr_node.nn_policy = softmax(curr_node.nn_policy);
    //valueは使わないはずだけど気分で
    curr_node.value = y.second[0];

    //詰み探索立ち上げ
    std::thread mate_thread(&SearcherForPlay::mateSearch, this, root, usi_option.draw_turn);

    //workerを立ち上げ
    std::vector<std::thread> threads(thread_num_);
    for (uint64_t i = 0; i < thread_num_; i++) {
        threads[i] = std::thread(&SearcherForPlay::parallelUctSearch, this, root, i);
    }

    //終了を待つ
    mate_thread.join();
    for (auto& t : threads) {
        t.join();
    }

    const auto& N = curr_node.N;

    printUSIInfo();
    if (usi_option.print_debug_info) {
        root.print();
        for (int32_t i = 0; i < curr_node.moves.size(); i++) {
            double nn_policy     = 100.0 * curr_node.nn_policy[i];
            double search_policy = 100.0 * N[i] / curr_node.sum_N;
#ifdef USE_CATEGORICAL
            double v = (N[i] > 0 ? expOfValueDist(curr_node.W[i]) / N[i] : MIN_SCORE);
#else
            double v = (N[i] > 0 ? curr_node.W[i] / N[i] : MIN_SCORE);
#endif
            printf("%3d  %5.1f  %5.1f  %+.3f  ", i, nn_policy, search_policy, v);
            curr_node.moves[i].printWithNewLine();
        }
    }

    //探索回数最大の手を選択する
    int32_t best_index = (int32_t)(std::max_element(N.begin(), N.end()) - N.begin());

    //選択した手の探索回数は少なくとも1以上であることを前提とする
    assert(N[best_index] != 0);

    //選択した着手の勝率の算出
#ifdef USE_CATEGORICAL
    auto best_wp = expOfValueDist(curr_node.W[best_index]) / N[best_index];
#else
    auto best_wp = curr_node.W[best_index] / N[best_index];
#endif

    //訪問回数に基づいた分布を得る
    std::vector<CalcType> distribution(curr_node.moves.size());
    for (int32_t i = 0; i < curr_node.moves.size(); i++) {
        distribution[i] = (CalcType)N[i] / curr_node.sum_N;
        assert(0.0 <= distribution[i] && distribution[i] <= 1.0);
    }

    //最善手
    Move best_move = (root.turn_number() < usi_option.random_turn ?
                      curr_node.moves[randomChoose(distribution)] :
                      curr_node.moves[best_index]);

    best_move.score = (Score)(best_wp);

    return best_move;
}

std::vector<Move> SearcherForPlay::getPV() const {
    std::vector<Move> pv;
    for (Index curr_node_index = current_root_index_; curr_node_index != UctHashTable::NOT_EXPANDED && !hash_table_[curr_node_index].moves.empty(); ) {
        const auto& N = hash_table_[curr_node_index].N;
        Index next_index = (int32_t)(std::max_element(N.begin(), N.end()) - N.begin());
        pv.push_back(hash_table_[curr_node_index].moves[next_index]);
        curr_node_index = hash_table_[curr_node_index].child_indices[next_index];
    }

    return pv;
}

void SearcherForPlay::printUSIInfo() const {
    const auto& curr_node = hash_table_[current_root_index_];

    //探索にかかった時間を求める
    auto finish_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_);

    auto selected_index = (int32_t)(std::max_element(curr_node.N.begin(), curr_node.N.end()) - curr_node.N.begin());

    //選択した着手の勝率の算出
#ifdef USE_CATEGORICAL
    double best_wp = expOfValueDist(curr_node.W[selected_index]) / curr_node.N[selected_index];
#else
    auto best_wp = (curr_node.N[selected_index] == 0 ? 0.0 : curr_node.W[selected_index] / curr_node.N[selected_index]);
#endif

#ifdef USE_CATEGORICAL
    //分布の表示
    constexpr int64_t gather_num = 3;
    for (int64_t i = 0; i < BIN_SIZE / gather_num; i++) {
        double p = 0.0;
        for (int64_t j = 0; j < gather_num; j++) {
            p += curr_node.W[selected_index][i * gather_num + j] / curr_node.N[selected_index];
        }
        printf("info string [%+6.2f:%06.2f%%]:", MIN_SCORE + VALUE_WIDTH * (gather_num * i + 1.5), p * 100);
        for (int64_t j = 0; j < p * 50; j++) {
            printf("*");
        }
        printf("\n");
    }
#endif

    int64_t ela = elapsed.count();
    printf("info nps %d time %d nodes %d hashfull %d score cp %d pv ",
           (int32_t)(curr_node.sum_N * 1000LL / std::max(ela, (int64_t)1)),
           (int32_t)(ela),
           curr_node.sum_N,
           (int32_t)(hash_table_.getUsageRate() * 1000),
           (int32_t)(best_wp * 1000));

    auto pv = getPV();
    for (auto m : pv) {
        std::cout << m << " ";
    }
    std::cout << std::endl;
}

void SearcherForPlay::parallelUctSearch(Position root, int32_t id) {
    //このスレッドに対するキューを参照で取る
    auto& input_queue = input_queues_[id];
    auto& index_queue = index_queues_[id];
    auto& route_queue = route_queues_[id];
    auto& action_queue = action_queues_[id];
    auto& redundancy_num = redundancy_num_[id];

    //限界に達するまで探索を繰り返す
    while (hash_table_[current_root_index_].sum_N < usi_option.search_limit && !shouldStop()) {
        //キューをクリア
        input_queue.clear();
        index_queue.clear();
        route_queue.clear();
        action_queue.clear();
        redundancy_num.clear();

        if (hash_table_[current_root_index_].sum_N >= next_print_node_num_) {
            printUSIInfo();
            next_print_node_num_ += usi_option.print_interval;
        }

        //評価要求を貯める
        for (uint64_t i = 0; i < search_batch_size_ && !shouldStop(); i++) {
            select(root, id);
        }

        //評価要求をGPUで計算
        if (!index_queue.empty()) {
            lock_expand_.lock();
            torch::NoGradGuard no_grad_guard;
            auto y = evaluator_->policyAndValueBatch(input_queue);
            lock_expand_.unlock();

            //書き込み
            for (int32_t i = 0; i < index_queue.size(); i++) {
                std::unique_lock<std::mutex> lock(lock_node_[index_queue[i]]);
                auto& curr_node = hash_table_[index_queue[i]];
                curr_node.nn_policy.resize(curr_node.moves.size());
                for (int32_t j = 0; j < curr_node.moves.size(); j++) {
                    curr_node.nn_policy[j] = y.first[i][curr_node.moves[j].toLabel()];
                }
                curr_node.nn_policy = softmax(curr_node.nn_policy);
                curr_node.value = y.second[i];
                curr_node.evaled = true;
            }
        }

        //バックアップ
        for (int32_t i = 0; i < route_queue.size(); i++) {
            backup(route_queue[i], action_queue[i], 1 - VIRTUAL_LOSS * redundancy_num[i]);
        }
    }
}

void SearcherForPlay::select(Position& pos, int32_t id) {
    std::stack<Index> curr_indices;
    std::stack<int32_t> curr_actions;

    auto index = current_root_index_;
    //ルートでは合法手が一つはあるはず
    assert(!hash_table_[index].moves.empty());

    //未展開の局面に至るまで遷移を繰り返す
    while(index != UctHashTable::NOT_EXPANDED) {
        std::unique_lock<std::mutex> lock(lock_node_[index]);

        if (hash_table_[index].moves.empty()) {
            //詰みの場合抜ける
            break;
        }

        Score repeat_score;
        if (index != current_root_index_ && pos.isRepeating(repeat_score)) {
            //繰り返しが発生している場合も抜ける
            break;
        }

        if (hash_table_[index].nn_policy.size() != hash_table_[index].moves.size()) {
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
        hash_table_[index].N[action] += VIRTUAL_LOSS;
        hash_table_[index].sum_N     += VIRTUAL_LOSS;

        //遷移
        pos.doMove(hash_table_[index].moves[action]);

        //index更新
        index = hash_table_[index].child_indices[action];
    }

    //expandNode内でこれらの情報は壊れる可能性があるので保存しておく
    index = curr_indices.top();
    auto action = curr_actions.top();
    auto move_num = curr_actions.size();

    //今の局面を展開・GPUに評価依頼を投げる
    auto leaf_index = expand(pos, curr_indices, curr_actions, id);

    //葉の直前ノードを更新
    lock_node_[index].lock();
    hash_table_[index].child_indices[action] = leaf_index;
    lock_node_[index].unlock();

    //バックアップはGPU計算後にやるので局面だけ戻す
    for (int32_t i = 0; i < move_num; i++) {
        pos.undo();
    }
}

Index SearcherForPlay::expand(Position& pos, std::stack<int32_t>& indices, std::stack<int32_t>& actions, int32_t id) {
    //全体をロック.このノードだけではなく探す部分も含めてなので
    std::unique_lock<std::mutex> lock(lock_expand_);

    auto index = hash_table_.findSameHashIndex(pos);

    // 合流先が検知できればそれを返す
    if (index != hash_table_.size()) {
        indices.push(index);
        if (hash_table_[index].evaled) {
            //評価済みならば,前回までのループでここへ違う経路で到達していたか,終端状態であるかのいずれか
            //どちらの場合でもバックアップして良い,と思う
            //GPUに送らないのでこのタイミングでバックアップを行う
            backup(indices, actions, 1 - VIRTUAL_LOSS);
        } else {
            //評価済みではないけどここへ到達したならば,同じループの中で同じ局面へ到達があったということ
            //全く同じ経路のものがあるかどうか確認
            auto itr = std::find(route_queues_[id].begin(), route_queues_[id].end(), indices);
            if (itr == route_queues_[id].end()) {
                //同じものがなかったならばバックアップ要求を追加
                route_queues_[id].push_back(indices);
                action_queues_[id].push_back(actions);
                redundancy_num_[id].push_back(1);
            } else {
                //同じものがあったならば重複数を増やして終わり
                redundancy_num_[id][itr - route_queues_[id].begin()]++;
            }
        }
        return index;
    }

    // 空のインデックスを探す
    index = hash_table_.searchEmptyIndex(pos);

    if (index == hash_table_.size()) {
        //置換表がいっぱいだったということ。どうしよう
    }

    //経路として記録
    indices.push(index);

    auto& curr_node = hash_table_[index];

    // 候補手の展開
    curr_node.moves = pos.generateAllMoves();
    curr_node.child_indices.assign(curr_node.moves.size(), UctHashTable::NOT_EXPANDED);
    curr_node.N.assign(curr_node.moves.size(), 0);
    curr_node.sum_N = 0;
    curr_node.evaled = false;
#ifdef USE_CATEGORICAL
    curr_node.W.assign(curr_node.moves.size(), {});
    curr_node.value = std::array<float, BIN_SIZE>{};
#else
    curr_node.W.assign(curr_node.moves.size(), 0.0);
    curr_node.value = 0.0;
#endif

    // ノードを評価
    Score repeat_score;
    if (pos.isRepeating(repeat_score)) {
        //繰り返し
#ifdef USE_CATEGORICAL
        curr_node.value = onehotDist(repeat_score);
#else
        curr_node.value = repeat_score;
#endif
        curr_node.evaled = true;
        //GPUに送らないのでこのタイミングでバックアップを行う
        backup(indices, actions, 1 - VIRTUAL_LOSS);
    } else if (curr_node.moves.empty()) {
        //打ち歩詰めなら勝ち,そうでないなら負け
        auto v = (pos.isLastMoveDropPawn() ? MAX_SCORE : MIN_SCORE);

#ifdef USE_CATEGORICAL
        curr_node.value = onehotDist(v);
#else
        curr_node.value = v;
#endif
        curr_node.evaled = true;
        //GPUに送らないのでこのタイミングでバックアップを行う
        backup(indices, actions, 1 - VIRTUAL_LOSS);
    } else {
        //GPUへ計算要求を投げる
        auto this_feature = pos.makeFeature();
        input_queues_[id].insert(input_queues_[id].end(), this_feature.begin(), this_feature.end());
        index_queues_[id].push_back(index);
        //バックアップ要求も投げる
        route_queues_[id].push_back(indices);
        action_queues_[id].push_back(actions);
        redundancy_num_[id].push_back(1);
    }

    return index;
}

void SearcherForPlay::backup(std::stack<int32_t>& indices, std::stack<int32_t>& actions, int32_t add_num) {
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

        //手番が変わるので反転
#ifdef USE_CATEGORICAL
        std::reverse(value.begin(), value.end());
#else
        value = MAX_SCORE + MIN_SCORE - value;
#endif
        // 探索結果の反映
        lock_node_[index].lock();
        hash_table_[index].W[action] += value;
        hash_table_[index].N[action] += add_num;
        hash_table_[index].sum_N     += add_num;
        assert(!hash_table_[index].moves.empty());
        lock_node_[index].unlock();
    }
}