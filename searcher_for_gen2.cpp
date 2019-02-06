#include "game_generator2.hpp"
#include "usi_options.hpp"
#include "operate_params.hpp"
#include <thread>
#include <stack>

Index
GameGenerator2::SearcherForGen2::expandNode(Position &pos, std::stack<int32_t> &indices, std::stack<int32_t> &actions) {
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
    // 千日手でも普通に展開して良いはず
    if (current_node.child_num > 0) {
        auto this_feature = pos.makeFeature();
        features_.resize(features_.size() + this_feature.size());
        std::copy(this_feature.begin(), this_feature.end(), features_.end() - this_feature.size());

        //現在のノードを追加
        indices.push(index);
        hash_indices_.push_back(indices);
        actions_.push_back(actions);
        ids_.push_back(id_);
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
        indices.push(index);
        backup(indices, actions);
    }

    return index;
}

bool GameGenerator2::SearcherForGen2::shouldGoNextPosition(){
    // 探索回数が最も多い手と次に多い手を求める
    int32_t max1 = 0, max2 = 0;
    for (auto e : hash_table_[current_root_index_].child_move_counts) {
        if (e > max1) {
            max2 = max1;
            max1 = e;
        } else if (e > max2) {
            max2 = e;
        }
    }

    // 残りの探索を全て次善手に費やしても最善手を超えられない場合は探索を打ち切る
    return (max1 - max2) > (usi_option.playout_limit - playout_num_);
}

std::vector<double> GameGenerator2::SearcherForGen2::dirichletDistribution(int32_t k, double alpha) {
    static std::random_device seed;
    static std::default_random_engine engine(seed());
    static constexpr double eps = 0.000000001;
    std::gamma_distribution<double> gamma(alpha, 1.0);
    std::vector<double> dirichlet(static_cast<unsigned long>(k));
    double sum = 0.0;
    for (int32_t i = 0; i < k; i++) {
        sum += (dirichlet[i] = std::max(gamma(engine), eps));
    }
    for (int32_t i = 0; i < k; i++) {
        dirichlet[i] /= sum;
    }
    return dirichlet;
}

int32_t GameGenerator2::SearcherForGen2::selectMaxUcbChild(const UctHashEntry & current_node) {
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

void GameGenerator2::SearcherForGen2::onePlay(Position &pos) {
    if (playout_num_++ == 0) {
        //初回の探索をする前にノイズを加える
        //Alpha Zeroの論文と同じディリクレノイズ
        auto& current_node = hash_table_[current_root_index_];
        constexpr double epsilon = 0.25;
        auto dirichlet = dirichletDistribution(current_node.child_num, 0.15);
        for (int32_t i = 0; i < current_node.child_num; i++) {
            current_node.nn_rates[i] = (CalcType) ((1.0 - epsilon) * current_node.nn_rates[i] + epsilon * dirichlet[i]);
        }
    }

    if (pos.generateAllMoves().empty()) {
        pos.print(false);
        assert(false);
    }

    std::stack<Index> curr_indices;
    std::stack<int32_t> curr_actions;

    auto index = current_root_index_;
    //ルートでは合法手が一つはあるはず
    assert(hash_table_[index].child_num != 0);

    //未展開の局面に至るまで遷移を繰り返す
    while(index != UctHashTable::NOT_EXPANDED) {
        if (hash_table_[index].child_num == 0) {
            //詰みの場合抜ける
            break;
        }

        //状態を記録
        curr_indices.push(index);

        //選択
        auto action = selectMaxUcbChild(hash_table_[index]);

        //取った行動を記録
        curr_actions.push(action);

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
    auto leaf_index = expandNode(pos, curr_indices, curr_actions);

    //葉の直前ノードを更新
    hash_table_[index].child_indices[action] = leaf_index;

    //バックアップはGPU計算後にやるので局面だけ戻す
    for (int32_t i = 0; i < move_num; i++) {
        pos.undo();
    }
}

bool GameGenerator2::SearcherForGen2::prepareForCurrPos(Position &root) {
    //古いハッシュを削除
    hash_table_.deleteOldHash(root, false);

    //ルートノードの展開
    std::stack<int32_t> indices;
    std::stack<int32_t> actions;
    current_root_index_ = expandNode(root, indices, actions);

    //探索回数を初期化
    playout_num_ = 0;

    //合法手が0かどうかを判定して返す
    return hash_table_[current_root_index_].child_num > 0;
}

std::pair<Move, TeacherType> GameGenerator2::SearcherForGen2::resultForCurrPos(Position &root) {
    const auto& current_node = hash_table_[current_root_index_];
    assert(current_node.child_num != 0);
    assert(current_node.move_count != 0);
    const auto& child_move_counts = current_node.child_move_counts;

    // 訪問回数最大の手を選択する
    int32_t best_index = (int32_t)(std::max_element(child_move_counts.begin(), child_move_counts.end()) - child_move_counts.begin());

    //選択した着手の勝率の算出
#ifdef USE_CATEGORICAL
    double best_wp = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        double v = current_node.child_wins[best_index][i] / child_move_counts[best_index];
        best_wp += VALUE_WIDTH * (0.5 + i) * v;
    }
#else
    auto best_wp = (child_move_counts[best_index] == 0 ? MIN_SCORE
                                                       : current_node.child_wins[best_index] / child_move_counts[best_index]);
#endif

    //投了しない場合教師データを作成
    TeacherType teacher;

    //valueのセット
#ifdef USE_CATEGORICAL
    teacher.value = valueToIndex(best_wp);
#else
    teacher.value = (CalcType)best_wp;
#endif

    //訪問回数に基づいた分布を得る
    std::vector<CalcType> distribution(static_cast<unsigned long>(current_node.child_num));
    assert(current_node.move_count == std::accumulate(child_move_counts.begin(), child_move_counts.end(), 0));
    for (int32_t i = 0; i < current_node.child_num; i++) {
        distribution[i] = (CalcType)child_move_counts[i] / current_node.move_count;
        assert(0 <= child_move_counts[i] && child_move_counts[i] <= current_node.move_count);
        if (!(0.0 <= distribution[i] && distribution[i] <= 1.0)) {
            std::cout << distribution[i] << std::endl;
            assert(false);
        }
    }

    //最善手
    Move best_move = (root.turn_number() < usi_option.random_turn ?
                      current_node.legal_moves[randomChoose(distribution)] :
                      current_node.legal_moves[best_index]);

    best_move.score = (Score)(best_wp);
    teacher.policy = best_move.toLabel();

    return { best_move, teacher };
}

void GameGenerator2::SearcherForGen2::backup(std::stack<int32_t> &indices, std::stack<int32_t> &actions) {
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
        hash_table_[index].child_wins[action] += value;
        hash_table_[index].move_count++;
        hash_table_[index].child_move_counts[action]++;
        assert(hash_table_[index].child_num != 0);
    }
}