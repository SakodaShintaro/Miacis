#include "searcher.hpp"

bool Searcher::stop_signal = false;

bool Searcher::shouldStop() {
    //シグナルのチェック
    if (Searcher::stop_signal) {
        return true;
    }

    //時間のチェック
    auto now_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now_time - start_);
    if (elapsed.count() >= time_limit_) {
        return true;
    }

    //ハッシュテーブルの容量チェック
    if (!hash_table_.hasEnoughSize()) {
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

int32_t Searcher::selectMaxUcbChild(const UctHashEntry& node) {
#ifdef USE_CATEGORICAL
    int32_t best_index = std::max_element(node.N.begin(), node.N.end()) - node.N.begin();
    FloatType best_value = expOfValueDist(hash_table_.QfromNextValue(node, best_index));
#endif

    int32_t max_index = -1;
    FloatType max_value = MIN_SCORE - 1;

    const int32_t sum = node.sum_N + node.virtual_sum_N;
    for (uint64_t i = 0; i < node.moves.size(); i++) {
        FloatType U = std::sqrt(sum + 1) / (node.N[i] + node.virtual_N[i] + 1);

#ifdef USE_CATEGORICAL
        FloatType P = 0.0;
        ValueType Q_dist = hash_table_.QfromNextValue(node, i);
        FloatType Q = expOfValueDist(Q_dist);
        for (int32_t j = std::min(valueToIndex(best_value) + 1, BIN_SIZE - 1); j < BIN_SIZE; j++) {
            P += Q_dist[j];
        }
        FloatType ucb = usi_options_.Q_coeff_x1000 / 1000.0 * Q
                      + usi_options_.C_PUCT_x1000 / 1000.0 * node.nn_policy[i] * U
                      + usi_options_.P_coeff_x1000 / 1000.0 * P;
#else
        FloatType Q = (node.N[i] == 0 ? MIN_SCORE : hash_table_.QfromNextValue(node, i));
        FloatType ucb = usi_options_.Q_coeff_x1000 / 1000.0 * Q
                      + usi_options_.C_PUCT_x1000 / 1000.0 * node.nn_policy[i] * U;
#endif

        if (ucb > max_value) {
            max_value = ucb;
            max_index = i;
        }
    }
    assert(0 <= max_index && max_index < (int32_t)node.moves.size());
    return max_index;
}

#ifdef SHOGI
bool Searcher::mateSearchForAttacker(Position& pos, int32_t depth) {
    assert(depth % 2 == 1);
    if (shouldStop()) {
        return false;
    }
    //全ての手を試してみる
    for (const auto& move : pos.generateAllMoves()) {
        pos.doMove(move);
        bool result = mateSearchForEvader(pos, depth - 1);
        pos.undo();
        if (result) {
            return true;
        }
    }
    return false;
}

bool Searcher::mateSearchForEvader(Position& pos, int32_t depth) {
    assert(depth % 2 == 0);
    if (shouldStop() || !pos.isChecked()) {
        return false;
    }

    if (depth == 0) {
        //詰みかつ打ち歩詰めでない
        return pos.generateAllMoves().empty() && !pos.isLastMoveDropPawn();
    }

    //全ての手を試してみる
    for (const auto& move : pos.generateAllMoves()) {
        pos.doMove(move);
        bool result = mateSearchForAttacker(pos, depth - 1);
        pos.undo();
        if (!result) {
            return false;
        }
    }

    //打ち歩詰めの確認
    return !pos.isLastMoveDropPawn();
}

void Searcher::mateSearch(Position pos, int32_t depth_limit) {
    auto& curr_node = hash_table_[hash_table_.root_index];
    for (int32_t depth = 1; !shouldStop() && depth <= depth_limit; depth += 2) {
        for (uint64_t i = 0; i < curr_node.moves.size(); i++) {
            pos.doMove(curr_node.moves[i]);
            bool result = mateSearchForEvader(pos, depth - 1);
            pos.undo();
            if (result) {
                //この手に書き込み
                //search_limitだけ足せば必ずこの手が選ばれるようになる
                curr_node.N[i]  += node_limit_;
                curr_node.sum_N += node_limit_;

                if (curr_node.child_indices[i] != UctHashTable::NOT_EXPANDED) {
#ifdef USE_CATEGORICAL
                    //普通の探索結果と値が混ざってしまいそう
                    //タイミングによっては問題が起こるかもしれない
                    hash_table_[curr_node.child_indices[i]].value[0] = 1;
                    for (int32_t j = 1; j < BIN_SIZE; j++) {
                        hash_table_[curr_node.child_indices[i]].value[j] = 0;
                    }
#else
                    hash_table_[curr_node.child_indices[i]].value = MIN_SCORE;
#endif
                }
                return;
            }
        }
    }
}
#endif

void Searcher::select(Position& pos) {
    std::stack<Index> curr_indices;
    std::stack<int32_t> curr_actions;

    Index index = hash_table_.root_index;
    //ルートでは合法手が一つはあるはず
    assert(!hash_table_[index].moves.empty());

    //未展開の局面に至るまで遷移を繰り返す
    while (index != UctHashTable::NOT_EXPANDED) {
        std::unique_lock<std::mutex> lock(hash_table_[index].mutex);

        if (pos.turnNumber() > usi_options_.draw_turn) {
            //手数が制限まで達している場合も抜ける
            break;
        }

        float score;
        if (index != hash_table_.root_index && pos.isFinish(score)) {
            //繰り返しが発生している場合も抜ける
            break;
        }

        if (hash_table_[index].nn_policy.size() != hash_table_[index].moves.size()) {
            //policyが展開されていなかったら抜ける
            //ここに来るのはこのselectループ内で先に展開されたがまだGPU計算が行われていないノードに達したとき
            break;
        }

        //状態を記録
        curr_indices.push(index);

        //選択
        auto action = selectMaxUcbChild(hash_table_[index]);

        //取った行動を記録
        curr_actions.push(action);

        //VIRTUAL_LOSSの追加
        hash_table_[index].virtual_N[action] += VIRTUAL_LOSS;
        hash_table_[index].virtual_sum_N += VIRTUAL_LOSS;

        //遷移
        pos.doMove(hash_table_[index].moves[action]);

        //index更新
        index = hash_table_[index].child_indices[action];
    }

    //expandNode内でこれらの情報は壊れる可能性があるので保存しておく
    index = curr_indices.top();
    int32_t action = curr_actions.top();
    uint64_t move_num = curr_actions.size();

    //今の局面を展開・GPUに評価依頼を投げる
    Index leaf_index = expand(pos, curr_indices, curr_actions);
    if (leaf_index == -1) {
        //shouldStopがtrueになったということ
        //基本的には置換表に空きがなかったということだと思われる
        return;
    }

    //葉の直前ノードを更新
    hash_table_[index].mutex.lock();
    hash_table_[index].child_indices[action] = leaf_index;
    hash_table_[index].mutex.unlock();

    //バックアップはGPU計算後にやるので局面だけ戻す
    for (uint64_t i = 0; i < move_num; i++) {
        pos.undo();
    }
}

Index Searcher::expand(Position& pos, std::stack<int32_t>& indices, std::stack<int32_t>& actions) {
    //置換表全体をロック
    hash_table_.mutex.lock();

    uint64_t index = hash_table_.findSameHashIndex(pos);

    //合流先が検知できればそれを返す
    if (index != hash_table_.size()) {
        //置換表全体のロックはもういらないので解放
        hash_table_.mutex.unlock();

        indices.push(index);
        if (hash_table_[index].evaled) {
            //評価済みならば,前回までのループでここへ違う経路で到達していたか,終端状態であるかのいずれか
            //どちらの場合でもバックアップして良い,と思う
            //GPUに送らないのでこのタイミングでバックアップを行う
            backup(indices, actions);
        } else {
            //評価済みではないけどここへ到達したならば,同じループの中で同じ局面へ到達があったということ
            //全く同じ経路のものがあるかどうか確認
            auto itr = std::find(backup_queue_.indices.begin(), backup_queue_.indices.end(), indices);
            if (itr == backup_queue_.indices.end()) {
                //同じものがなかったならばバックアップ要求を追加
                backup_queue_.indices.push_back(indices);
                backup_queue_.actions.push_back(actions);
            }
        }
        return index;
    }

    //空のインデックスを探す
    index = hash_table_.searchEmptyIndex(pos);

    //経路として記録
    indices.push(index);

    //テーブル全体を使うのはここまで
    hash_table_.mutex.unlock();

    //空のインデックスが見つからなかった
    if (index == hash_table_.size()) {
        return -1;
    }

    //取得したノードをロック
    hash_table_[index].mutex.lock();

    UctHashEntry& curr_node = hash_table_[index];

    // 候補手の展開
    curr_node.moves = pos.generateAllMoves();
    curr_node.moves.shrink_to_fit();
    curr_node.child_indices.assign(curr_node.moves.size(), UctHashTable::NOT_EXPANDED);
    curr_node.child_indices.shrink_to_fit();
    curr_node.N.assign(curr_node.moves.size(), 0);
    curr_node.N.shrink_to_fit();
    curr_node.virtual_N.assign(curr_node.moves.size(), 0);
    curr_node.virtual_N.shrink_to_fit();
    curr_node.sum_N = 0;
    curr_node.virtual_sum_N = 0;
    curr_node.evaled = false;
    curr_node.value = ValueType{};

    //ノードを評価
    float finish_score;
    if (pos.isFinish(finish_score)) {
#ifdef USE_CATEGORICAL
        curr_node.value = onehotDist(finish_score);
#else
        curr_node.value = finish_score;
#endif
        curr_node.evaled = true;
        //GPUに送らないのでこのタイミングでバックアップを行う
        hash_table_[index].mutex.unlock();
        backup(indices, actions);
    } else if (pos.turnNumber() > usi_options_.draw_turn) {
        FloatType value = (MAX_SCORE + MIN_SCORE) / 2;
#ifdef USE_CATEGORICAL
        curr_node.value = onehotDist(value);
#else
        curr_node.value = value;
#endif
        curr_node.evaled = true;
        //GPUに送らないのでこのタイミングでバックアップを行う
        hash_table_[index].mutex.unlock();
        backup(indices, actions);
    } else {
        hash_table_[index].mutex.unlock();

        //GPUへの計算要求を追加
        std::vector<FloatType> this_feature = pos.makeFeature();
        gpu_queue_.inputs.insert(gpu_queue_.inputs.end(), this_feature.begin(), this_feature.end());
        gpu_queue_.hash_tables.emplace_back(hash_table_);
        gpu_queue_.indices.push_back(index);
        //バックアップ要求も追加
        backup_queue_.indices.push_back(indices);
        backup_queue_.actions.push_back(actions);
    }

    return index;
}

void Searcher::backup(std::stack<int32_t>& indices, std::stack<int32_t>& actions) {
    assert(actions.empty() || indices.size() == actions.size() + 1);

    auto leaf = indices.top();
    indices.pop();
    hash_table_[leaf].mutex.lock();
    auto value = hash_table_[leaf].value;
    hash_table_[leaf].mutex.unlock();

    //毎回計算するのは無駄だけど仕方ないか
    //FloatType lambda = usi_options_.UCT_lambda_x1000 / 1000.0;
    static constexpr FloatType lambda = 1.0;

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
        hash_table_[index].mutex.lock();

        UctHashEntry& node = hash_table_[index];

        //探索回数の更新
        node.N[action]++;
        node.sum_N++;
        node.virtual_sum_N -= node.virtual_N[action];
        node.virtual_N[action] = 0;

        //価値の更新
        ValueType curr_v = node.value;
        FloatType alpha = 1.0f / (node.sum_N + 1);
        node.value += alpha * (value - curr_v);
        value = lambda * value + (1.0f - lambda) * curr_v;

        //最大バックアップ
//#ifdef USE_CATEGORICAL
//        node.value = onehotDist(MIN_SCORE);
//        for (int32_t i = 0; i < node.moves.size(); i++) {
//            auto q = QfromNextValue(node, i);
//            if (expOfValueDist(q) > expOfValueDist(node.value)) {
//                node.value = q;
//            }
//        }
//#else
//        node.value = MIN_SCORE;
//        for (int32_t i = 0; i < node.moves.size(); i++) {
//            node.value = std::max(node.value, QfromNextValue(node, i));
//        }
//#endif

        hash_table_[index].mutex.unlock();
    }
}

void Searcher::backupAll() {
    for (uint64_t i = 0; i < backup_queue_.indices.size(); i++) {
        backup(backup_queue_.indices[i], backup_queue_.actions[i]);
    }
    backup_queue_.indices.clear();
    backup_queue_.actions.clear();
}