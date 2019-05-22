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
    int32_t search_num = hash_table_[current_root_index_].sum_N + hash_table_[current_root_index_].virtual_sum_N;
    return search_num >= node_limit_;
}

int32_t Searcher::selectMaxUcbChild(const UctHashEntry& current_node) {
#ifdef USE_CATEGORICAL
    int32_t selected_index = -1, max_num = -1;
    for (int32_t i = 0; i < current_node.moves.size(); i++) {
        int32_t num = current_node.N[i] + current_node.virtual_N[i];
        if (num > max_num) {
            selected_index = i;
            max_num = num;
        }
    }
    double best_wp = expOfValueDist(current_node.Q[selected_index]) / max_num;
#endif

    // ucb = Q(s, a) + U(s, a)
    // Q(s, a) = W(s, a) / N(s, a)
    // U(s, a) = C * P(s, a) * sqrt(sum_b(B(s, b)) / (1 + N(s, a))

    constexpr double C_base = 19652.0;
    constexpr double C_init = 1.25;

    int32_t max_index = -1;
    double max_value = MIN_SCORE - 1;

    const int32_t sum = current_node.sum_N + current_node.virtual_sum_N;
    for (int32_t i = 0; i < current_node.moves.size(); i++) {
        auto visit_num = current_node.N[i] + current_node.virtual_N[i];

#ifdef USE_CATEGORICAL
        double Q;
        if (visit_num == 0) {
            //中間を初期値とする
            Q = (MAX_SCORE + MIN_SCORE) / 2;
        } else {
            Q = 0.0;
            for (int32_t j = std::min(valueToIndex(best_wp) + 1, BIN_SIZE - 1); j < BIN_SIZE; j++) {
                Q += current_node.Q[i][j] * current_node.N[i] / visit_num;
            }
        }
#else
        double Q = (visit_num == 0 ? (MAX_SCORE + MIN_SCORE) / 2 : current_node.Q[i] * current_node.N[i] / visit_num);
#endif
        double U = std::sqrt(sum + 1) / (visit_num + 1);
        double C = (std::log((sum + C_base + 1) / C_base) + C_init);
        double ucb = Q + C * current_node.nn_policy[i] * U;

        if (ucb > max_value) {
            max_value = ucb;
            max_index = i;
        }
    }
    assert(0 <= max_index && max_index < (int32_t)current_node.moves.size());
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
    auto& curr_node = hash_table_[current_root_index_];
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
#ifdef USE_CATEGORICAL
                curr_node.Q[i][BIN_SIZE - 1] += 1;
#else
                curr_node.Q[i] += MAX_SCORE;
#endif
                return;
            }
        }
    }
}