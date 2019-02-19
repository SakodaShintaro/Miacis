#ifndef MIACIS_SEARCHER_COMMON_HPP
#define MIACIS_SEARCHER_COMMON_HPP

#include"operate_params.hpp"

inline ValueType reverse(ValueType value) {
#ifdef USE_CATEGORICAL
    //カテゴリカルなら反転を返す
    std::reverse(value.begin(), value.end());
    return value;
#else
    return MAX_SCORE + MIN_SCORE - value;
#endif
}

inline int32_t selectMaxUcbChild(const UctHashEntry& current_node) {
    const auto& N = current_node.N;

#ifdef USE_CATEGORICAL
    int32_t selected_index = (int32_t)(std::max_element(N.begin(), N.end()) - N.begin());
    double best_wp = expOfValueDist(current_node.W[selected_index]) / N[selected_index];
#endif

    // ucb = Q(s, a) + U(s, a)
    // Q(s, a) = W(s, a) / N(s, a)
    // U(s, a) = C_PUCT * P(s, a) * sqrt(sum_b(B(s, b)) / (1 + N(s, a))
    constexpr double C_PUCT = 1.0;

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
        double U = std::sqrt(current_node.move_count + 1) / (N[i] + 1);
        double ucb = Q + C_PUCT * current_node.nn_rates[i] * U;

        if (ucb > max_value) {
            max_value = ucb;
            max_index = i;
        }
    }
    assert(0 <= max_index && max_index < (int32_t)current_node.moves.size());
    return max_index;
}

#endif //MIACIS_SEARCHER_COMMON_HPP