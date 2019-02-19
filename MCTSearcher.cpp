#include"MCTSearcher.hpp"
#include"searcher_common.hpp"
#include"operate_params.hpp"
#include"usi_options.hpp"
#include<stack>
#include<iomanip>

Move MCTSearcher::think(Position& root) {
    //思考開始時間をセット
    start_ = std::chrono::steady_clock::now();

    //古いハッシュを削除
    hash_table_.deleteOldHash(root, false);

    //ルートノードの展開
    current_root_index_ = expandNode(root);
    auto& current_node = hash_table_[current_root_index_];

    //合法手が0だったら投了
    if (current_node.moves.empty()) {
        return NULL_MOVE;
    }

    //探索を繰り返す.探索回数が閾値に達する,または打ち切り判定がtrueになったらループを抜ける
    for (int32_t i = 0; i < usi_option.playout_limit; i++) {
        //1回探索
        uctSearch(root, current_root_index_);

        //再帰じゃない探索.なぜか遅くなった
        //onePlay(root);

        //探索を打ち切るか確認
        if (shouldStop() || !hash_table_.hasEnoughSize()) {
            break;
        }
    }

    const auto& N = current_node.N;

    printUSIInfo();
    root.print(true);
    for (int32_t i = 0; i < current_node.moves.size(); i++) {
        double nn = 100.0 * current_node.nn_policy[i];
        double p  = 100.0 * N[i] / current_node.sum_N;
#ifdef USE_CATEGORICAL
        double v = (N[i] > 0 ? expOfValueDist(current_node.W[i]) / N[i] : MIN_SCORE);
#else
        double v = (N[i] > 0 ? current_node.W[i] / N[i] : MIN_SCORE);
#endif
        printf("%3d  %4.1f  %4.1f  %+.3f  ", i, nn, p, v);
        current_node.moves[i].print();
    }

    //探索回数最大の手を選択する
    int32_t best_index = (int32_t)(std::max_element(N.begin(), N.end()) - N.begin());

    //選んだ手の探索回数は少なくとも1以上であることを前提とする
    assert(N[best_index] != 0);

    //選択した着手の勝率の算出
#ifdef USE_CATEGORICAL
    double best_wp = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        double v = current_node.W[best_index][i] / N[best_index];
        best_wp += VALUE_WIDTH * (0.5 + i) * v;
    }
#else
    auto best_wp = current_node.W[best_index] / N[best_index];
#endif

    //best_moveの選択
    Move best_move;
    if (root.turn_number() >= usi_option.random_turn) {
        //最善手をそのまま返す
        best_move = current_node.moves[best_index];
    } else {
        //探索回数を正規化して分布を得る
        std::vector<CalcType> distribution(current_node.moves.size());
        for (int32_t i = 0; i < current_node.moves.size(); i++) {
            distribution[i] = (CalcType)N[i] / current_node.sum_N;
            assert(0.0 <= distribution[i] && distribution[i] <= 1.0);
        }

        //分布からランダムにサンプリング
        best_move = current_node.moves[randomChoose(distribution)];
    }

    //使うかどうかは微妙だけどscoreに最善の値をセットする
    best_move.score = best_wp;

    return best_move;
}

ValueType MCTSearcher::uctSearch(Position & pos, Index current_index) {
    auto& current_node = hash_table_[current_index];

    if (current_node.moves.empty()) {
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

    // UCB値が最大の手を求める
    auto next_index = selectMaxUcbChild(current_node);

    // 選んだ手を着手
    pos.doMove(current_node.moves[next_index]);

    ValueType result;
    Score score;
    // ノードの展開の確認
    if (pos.isRepeating(score)) {
#ifdef USE_CATEGORICAL
        result = onehotDist(score);
#else
        result = score;
#endif
    } else if (child_indices[next_index] == UctHashTable::NOT_EXPANDED) {
        // ノードの展開
        auto index = expandNode(pos);
        child_indices[next_index] = index;
        result = hash_table_[index].value;
    } else {
        // 手番を入れ替えて1手深く読む
        result = uctSearch(pos, child_indices[next_index]);
    }
    //手番が変わっているので反転
    result = reverse(result);

    // 探索結果の反映
    current_node.sum_N++;
    current_node.W[next_index] += result;
    current_node.N[next_index]++;

    // 手を戻す
    pos.undo();

    return result;
}

Index MCTSearcher::expandNode(Position& pos) {
    auto index = hash_table_.findSameHashIndex(pos);

    // 合流先が検知できればそれを返す
    if (index != hash_table_.size()) {
        return index;
    }

    // 空のインデックスを探す
    index = hash_table_.searchEmptyIndex(pos);

    auto& current_node = hash_table_[index];

    // 候補手の展開
    current_node.moves = pos.generateAllMoves();
    current_node.child_indices = std::vector<int32_t>(current_node.moves.size(), UctHashTable::NOT_EXPANDED);
    current_node.N = std::vector<int32_t>(current_node.moves.size(), 0);

    // 現在のノードの初期化
    current_node.sum_N = 0;
    current_node.evaled = false;
#ifdef USE_CATEGORICAL
    current_node.W = std::vector<std::array<CalcType, BIN_SIZE>>(current_node.moves.size());
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        current_node.value[i] = 0.0;
        for (int32_t j = 0; j < current_node.moves.size(); j++) {
            current_node.W[j][i] = 0.0;
        }
    }
#else
    current_node.value = 0.0;
    current_node.W = std::vector<float>(current_node.moves.size(), 0.0);
#endif

    // ノードを評価
    if (!current_node.moves.empty()) {
        evalNode(pos, index);
    } else {
        //打ち歩詰めなら勝ち,そうでないなら負け
        auto v = (pos.isLastMoveDropPawn() ? MAX_SCORE : MIN_SCORE);

#ifdef USE_CATEGORICAL
        current_node.value = onehotDist(v);
#else
        current_node.value = v;
#endif
        current_node.evaled = true;
    }

    return index;
}

void MCTSearcher::evalNode(Position& pos, Index index) {
    auto& current_node = hash_table_[index];
    std::vector<float> legal_move_policy(current_node.moves.size());

#ifdef USE_LIBTORCH
    auto policy_and_value = evaluator_->policyAndValue(pos);
#else
    auto policy_and_value = evaluator_.policyAndValue(pos);
#endif

    for (int32_t i = 0; i < current_node.moves.size(); i++) {
        legal_move_policy[i] = policy_and_value.first[current_node.moves[i].toLabel()];
    }

    //ノードの値を計算
    current_node.value = policy_and_value.second;

    //softmax分布にする
    current_node.nn_policy = softmax(legal_move_policy);

    Score repeat_score;
    if (pos.isRepeating(repeat_score)) {
#ifdef USE_CATEGORICAL
        current_node.value = onehotDist(sigmoid(repeat_score, CP_GAIN));
#else
        current_node.value = repeat_score;
#endif
    }

    current_node.evaled = true;
}

bool MCTSearcher::isTimeOver() {
    //時間のチェック
    auto now_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now_time - start_);
    return (elapsed.count() >= usi_option.limit_msec - usi_option.byoyomi_margin);
}

bool MCTSearcher::shouldStop() {
    return isTimeOver();
//    if (isTimeOver()) {
//        return true;
//    }
//    return false;
//
//    // 探索回数が最も多い手と次に多い手を求める
//    int32_t max1 = 0, max2 = 0;
//    for (auto e : hash_table_[current_root_index_].N) {
//        if (e > max1) {
//            max2 = max1;
//            max1 = e;
//        } else if (e > max2) {
//            max2 = e;
//        }
//    }
//
//    // 残りの探索を全て次善手に費やしても最善手を超えられない場合は探索を打ち切る
//    return (max1 - max2) > (usi_option.playout_limit - playout_num);
}

std::vector<Move> MCTSearcher::getPV() const {
    std::vector<Move> pv;
    for (Index curr_node_index = current_root_index_; curr_node_index != UctHashTable::NOT_EXPANDED && !hash_table_[curr_node_index].moves.empty(); ) {
        const auto& child_move_counts = hash_table_[curr_node_index].N;
        Index next_index = (int32_t)(std::max_element(child_move_counts.begin(), child_move_counts.end()) - child_move_counts.begin());
        pv.push_back(hash_table_[curr_node_index].moves[next_index]);
        curr_node_index = hash_table_[curr_node_index].child_indices[next_index];
    }

    return pv;
}

void MCTSearcher::printUSIInfo() const {
    const auto& current_node = hash_table_[current_root_index_];

    //探索にかかった時間を求める
    auto finish_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_);

    const auto& N = current_node.N;
    auto selected_index = (int32_t)(std::max_element(N.begin(), N.end()) - N.begin());

    //選択した着手の勝率の算出
#ifdef USE_CATEGORICAL
    double best_wp = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        double v = current_node.W[selected_index][i] / N[selected_index];
        best_wp += VALUE_WIDTH * (0.5 + i) * v;
    }
#else
    auto best_wp = (N[selected_index] == 0 ? 0.0
                                                           : current_node.W[selected_index] / N[selected_index]);
#endif

    //勝率を評価値に変換
    //int32_t cp = inv_sigmoid(best_wp, CP_GAIN);
    int32_t cp = (int32_t)(best_wp * 1000);

    printf("info nps %d time %d nodes %d hashfull %d score cp %d pv ",
           (int)(current_node.sum_N * 1000 / std::max((long long)elapsed.count(), 1LL)),
           (int)(elapsed.count()),
           current_node.sum_N,
           (int)(hash_table_.getUsageRate() * 1000),
           cp);

    auto pv = getPV();
    for (auto m : pv) {
        std::cout << m << " ";
    }
    std::cout << std::endl;
}

void MCTSearcher::onePlay(Position &pos) {
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
        pos.doMove(hash_table_[index].moves[action]);

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
        hash_table_[index].W[action] += result;
        hash_table_[index].sum_N++;
        hash_table_[index].N[action]++;
    }
}