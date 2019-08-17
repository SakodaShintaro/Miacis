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

    int32_t search_num = hash_table_[root_index_].sum_N + hash_table_[root_index_].virtual_sum_N;
    return search_num >= node_limit_;
}

int32_t Searcher::selectMaxUcbChild(const UctHashEntry& node) {
#ifdef USE_CATEGORICAL
    int32_t best_index = std::max_element(node.N.begin(), node.N.end()) - node.N.begin();
    FloatType best_value = expOfValueDist(QfromNextValue(node, best_index));
#endif

    int32_t max_index = -1;
    FloatType max_value = MIN_SCORE - 1;

    const int32_t sum = node.sum_N + node.virtual_sum_N;
    for (uint64_t i = 0; i < node.moves.size(); i++) {
        FloatType U = std::sqrt(sum + 1) / (node.N[i] + node.virtual_N[i] + 1);

#ifdef USE_CATEGORICAL
        FloatType P = 0.0;
        ValueType Q_dist = QfromNextValue(node, i);
        FloatType Q = expOfValueDist(Q_dist);
        for (int32_t j = std::min(valueToIndex(best_value) + 1, BIN_SIZE - 1); j < BIN_SIZE; j++) {
            P += Q_dist[j];
        }
        FloatType ucb = Q_coeff_ * Q + C_PUCT_ * node.nn_policy[i] * U + P_coeff_ * P;
#else
        FloatType Q = (node.N[i] == 0 ? (MAX_SCORE + MIN_SCORE) / 2 : QfromNextValue(node, i));
        FloatType ucb = Q_coeff_ * Q + C_PUCT_ * node.nn_policy[i] * U;
#endif

        if (ucb > max_value) {
            max_value = ucb;
            max_index = i;
        }
    }
    assert(0 <= max_index && max_index < (int32_t)node.moves.size());
    return max_index;
}

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
    auto& curr_node = hash_table_[root_index_];
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

ValueType Searcher::QfromNextValue(const UctHashEntry& node, int32_t i) const {
#ifdef USE_CATEGORICAL
    if (node.child_indices[i] == UctHashTable::NOT_EXPANDED) {
        return onehotDist(MIN_SCORE);
    }
    auto v = hash_table_[node.child_indices[i]].value;
    std::reverse(v.begin(), v.end());
    return v;
#else
    if (node.child_indices[i] == UctHashTable::NOT_EXPANDED) {
        return MIN_SCORE;
    }
    return MAX_SCORE + MIN_SCORE - hash_table_[node.child_indices[i]].value;
#endif
}