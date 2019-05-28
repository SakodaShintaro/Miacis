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
    int32_t search_num = hash_table_[root_index_].sum_N + hash_table_[root_index_].virtual_sum_N;
    return search_num >= node_limit_;
}

int32_t Searcher::selectMaxUcbChild(const UctHashEntry& node) {
#ifdef USE_CATEGORICAL
    int32_t best_index = -1, max_num = -1;
    for (int32_t i = 0; i < node.moves.size(); i++) {
        int32_t num = node.N[i] + node.virtual_N[i];
        if (num > max_num) {
            best_index = i;
            max_num = num;
        }
    }
    double best_wp = (node.child_indices[best_index] == -1 ? MIN_SCORE : expOfValueDist(QfromNextValue(node, best_index)));
#endif

    constexpr double C_PUCT = 2.5;

    int32_t max_index = -1;
    double max_value = MIN_SCORE - 1;

    const int32_t sum = node.sum_N + node.virtual_sum_N;
    for (int32_t i = 0; i < node.moves.size(); i++) {
        const int32_t visit_num = node.N[i] + node.virtual_N[i];

#ifdef USE_CATEGORICAL
        double Q;
        if (visit_num == 0) {
            //中間を初期値とする
            Q = (MAX_SCORE + MIN_SCORE) / 2;
        } else {
            Q = 0.0;
            ValueType Q_dist{};
            if (node.child_indices[i] != UctHashTable::NOT_EXPANDED) {
                Q_dist = hash_table_[node.child_indices[i]].value;
            }
            std::reverse(Q_dist.begin(), Q_dist.end());
            for (int32_t j = std::min(valueToIndex(best_wp) + 1, BIN_SIZE - 1); j < BIN_SIZE; j++) {
                Q += Q_dist[j];
            }
        }
#else
        double Q = (node.N[i] == 0 ? (MAX_SCORE + MIN_SCORE) / 2 : QfromNextValue(node, i));
#endif
        double U = std::sqrt(sum + 1) / (visit_num + 1);
        double ucb = Q + C_PUCT * node.nn_policy[i] * U;

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
        for (int32_t i = 0; i < curr_node.moves.size(); i++) {
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
    assert(node.child_indices[i] != UctHashTable::NOT_EXPANDED);
#ifdef USE_CATEGORICAL
    auto v = hash_table_[node.child_indices[i]].value;
    std::reverse(v.begin(), v.end());
    return v;
#else
    return -hash_table_[node.child_indices[i]].value;
#endif
}