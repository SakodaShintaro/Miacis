#include "searcher.hpp"
#include "usi_options.hpp"
#include "operate_params.hpp"

bool Searcher::shouldStop() {
    //時間のチェック
    auto now_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now_time - start_);
    if (elapsed.count() >= usi_option.limit_msec - usi_option.byoyomi_margin) {
        return true;
    }

    if (!hash_table_.hasEnoughSize()) {
        return true;
    }

    //探索回数が最も多い手と2番目に多い手を求める
    int32_t max1 = 0, max2 = 0;
    for (auto e : hash_table_[current_root_index_].N) {
        if (e > max1) {
            max2 = max1;
            max1 = e;
        } else if (e > max2) {
            max2 = e;
        }
    }

    // 残りの探索を全て次善手に費やしても最善手を超えられない場合は探索を打ち切る
    return (max1 - max2) >= (usi_option.search_limit - hash_table_[current_root_index_].sum_N);
}

int32_t Searcher::selectMaxUcbChild(const UctHashEntry& current_node) {
    const auto& N = current_node.N;

#ifdef USE_CATEGORICAL
    int32_t selected_index = (int32_t)(std::max_element(N.begin(), N.end()) - N.begin());
    double best_wp = expOfValueDist(current_node.W[selected_index]) / N[selected_index];
#endif

    // ucb = Q(s, a) + U(s, a)
    // Q(s, a) = W(s, a) / N(s, a)
    // U(s, a) = C_PUCT * P(s, a) * sqrt(sum_b(B(s, b)) / (1 + N(s, a))
    constexpr double C_PUCT = 1.0;

//    constexpr double C_base = 19652.0;
//    constexpr double C_init = 1.25;

    int32_t max_index = -1;
    double max_value = MIN_SCORE - 1;
    for (int32_t i = 0; i < current_node.moves.size(); i++) {
#ifdef USE_CATEGORICAL
        double Q;
        if (N[i] == 0) {
            //中間を初期値とする
            Q = (MAX_SCORE + MIN_SCORE) / 2;
        } else {
            Q = 0.0;
            for (int32_t j = std::min((int32_t)(best_wp * BIN_SIZE) + 1, BIN_SIZE - 1); j < BIN_SIZE; j++) {
                Q += current_node.W[i][j] / N[i];
            }
        }
#else
        double Q = (N[i] == 0 ? (MAX_SCORE + MIN_SCORE) / 2 : current_node.W[i] / N[i]);
#endif
        double U = std::sqrt(current_node.sum_N + 1) / (N[i] + 1);
        double ucb = Q + C_PUCT * current_node.nn_policy[i] * U;

        //double C = (std::log((current_node.sum_N + C_base + 1) / C_base) + C_init);
        //double ucb = Q + C * current_node.policy[i] * U;

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
                //playout_limitだけ足せば必ずこの手が選ばれるようになる
                curr_node.N[i]  += usi_option.search_limit;
                curr_node.sum_N += usi_option.search_limit;
#ifdef USE_CATEGORICAL
                curr_node.value[BIN_SIZE - 1] += usi_option.search_limit;
#else
                curr_node.value += MAX_SCORE * usi_option.search_limit;
#endif
                return;
            }
        }
    }
}