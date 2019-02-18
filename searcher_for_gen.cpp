#include "game_generator.hpp"
#include "searcher_common.hpp"
#include "usi_options.hpp"
#include "operate_params.hpp"
#include <thread>
#include <stack>
#include <iomanip>

//共通するものは分岐する前に置いておく
std::vector<double> GameGenerator::SearcherForGen::dirichletDistribution(int32_t k, double alpha) {
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

bool GameGenerator::SearcherForGen::shouldStop() {
    if (usi_option.stop_signal) {
        //停止信号が来ていたら問答無用で止まる
        return true;
    }

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

#ifdef USE_PARALLEL_SEARCHER

std::pair<Move, TeacherType> GameGenerator::SearcherForGen::think(Position& root) {
    //古いハッシュを削除
    hash_table_.deleteOldHash(root, false);

    //ルートノードの展開
    current_root_index_ = expandNode(root);
    auto& current_node = hash_table_[current_root_index_];

    //合法手が0だったら投了
    if (current_node.child_num == 0) {
        return { NULL_MOVE, TeacherType() };
    }

    while (!hash_table_[current_root_index_].evaled) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }

    //ノイズを加える
    //Alpha Zeroの論文と同じディリクレノイズ
    constexpr double epsilon = 0.25;
    auto dirichlet = dirichletDistribution(current_node.child_num, 0.15);
    for (int32_t i = 0; i < current_node.child_num; i++) {
        current_node.nn_rates[i] = (CalcType) ((1.0 - epsilon) * current_node.nn_rates[i] + epsilon * dirichlet[i]);
    }

    //初期化
    playout_num_ = 0;

    //ここで探索
    //プレイアウトを繰り返す
    //探索回数が閾値を超える、または探索が打ち切られたらループを抜ける
    while (playout_num_ < usi_option.playout_limit) {
        //探索回数を1回増やす
        playout_num_++;

        //1回プレイアウト
        uctSearch(root, current_root_index_);

        //onePlay(root);

        assert(hash_table_.hasEnoughSize());

        //探索を打ち切るか確認
        if (shouldStop()) {
            break;
        }
    }

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
    auto best_wp = (child_move_counts[best_index] == 0 ? 0.0
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
    for (int32_t i = 0; i < current_node.child_num; i++) {
        distribution[i] = (CalcType)child_move_counts[i] / current_node.move_count;
        assert(0.0 <= distribution[i] && distribution[i] <= 1.0);
    }

    //最善手
    Move best_move = (root.turn_number() < usi_option.random_turn ?
                      current_node.legal_moves[randomChoose(distribution)] :
                      current_node.legal_moves[best_index]);

    best_move.score = (Score)(best_wp);
    teacher.policy = best_move.toLabel();

    return { best_move, teacher };
}

ValueType GameGenerator::SearcherForGen::uctSearch(Position & pos, Index current_index) {
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

    // UCB値が最大の手を求める
    auto next_index = selectMaxUcbChild(current_node);

    // 選んだ手を着手
    pos.doMove(current_node.legal_moves[next_index]);

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
        while (!hash_table_[index].evaled) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }

        result = hash_table_[index].value;
    } else {
        // 手番を入れ替えて1手深く読む
        result = uctSearch(pos, child_indices[next_index]);
    }
    //手番が変わっているので反転
    result = reverse(result);

    // 探索結果の反映
    current_node.move_count++;
    current_node.child_wins[next_index] += result;
    current_node.child_move_counts[next_index]++;

    // 手を戻す
    pos.undo();

    return result;
}

Index GameGenerator::SearcherForGen::expandNode(Position& pos) {
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
        gg_.gpu_mutex.lock();
        gg_.current_features_.resize(gg_.current_features_.size() + this_feature.size());
        std::copy(this_feature.begin(), this_feature.end(), gg_.current_features_.end() - this_feature.size());
        gg_.current_hash_index_queue_.push_back(index);
        gg_.current_thread_ids_.push_back(id_);
        gg_.gpu_mutex.unlock();
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

void GameGenerator::SearcherForGen::onePlay(Position &pos) {
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

#else

Index GameGenerator::SearcherForGen::expandNode(Position& pos, std::stack<int32_t>& indices, std::stack<int32_t>& actions) {
    auto index = hash_table_.findSameHashIndex(pos.hash_value(), static_cast<int16_t>(pos.turn_number()));

    // 合流先が検知できればそれを返す
    if (index != hash_table_.size()) {
        //GPUに送らないのでこのタイミングでバックアップを行う
        indices.push(index);
        backup(indices, actions);
        return index;
    }

    // 空のインデックスを探す
    index = hash_table_.searchEmptyIndex(pos.hash_value(), static_cast<int16_t>(pos.turn_number()));

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
//    Score repeat_score;
//    if (pos.isRepeating(repeat_score)) {
//        //繰り返し
//#ifdef USE_CATEGORICAL
//        current_node.value = onehotDist(repeat_score);
//#else
//        current_node.value = repeat_score;
//#endif
//        current_node.evaled = true;
//        //GPUに送らないのでこのタイミングでバックアップを行う
//        indices.push(index);
//        backup(indices, actions);
//    } else
    if (current_node.child_num > 0) {
        //特徴量の追加
        auto this_feature = pos.makeFeature();
        features_.insert(features_.end(), this_feature.begin(), this_feature.end());

        //インデックス,行動の履歴およびidを追加
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

void GameGenerator::SearcherForGen::onePlay(Position &pos) {
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

//        Score repeat_score;
//        if (index != current_root_index_ && pos.isRepeating(repeat_score)) {
//            //繰り返しが発生している場合も抜ける
//            break;
//        }

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

bool GameGenerator::SearcherForGen::prepareForCurrPos(Position &root) {
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

std::pair<Move, TeacherType> GameGenerator::SearcherForGen::resultForCurrPos(Position &root) {
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
    std::vector<CalcType> distribution(static_cast<uint64_t>(current_node.child_num));
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

void GameGenerator::SearcherForGen::backup(std::stack<int32_t> &indices, std::stack<int32_t> &actions) {
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

void GameGenerator::SearcherForGen::mateSearch(Position pos, int32_t depth) {
    assert(depth % 2 == 1);
    auto& curr_node = hash_table_[current_root_index_];
    for (int32_t i = 0; i < curr_node.child_num; i++) {
        pos.doMove(curr_node.legal_moves[i]);
        bool result = mateSearchForEvader(pos, depth - 1);
        pos.undo();
        if (result) {
            //この手に書き込み
            //playout_limitだけ足せば必ずこの手が選ばれるようになる
            curr_node.child_move_counts[i] += usi_option.playout_limit;
        }
    }
}

bool GameGenerator::SearcherForGen::mateSearchForAttacker(Position& pos, int32_t depth) {
    assert(depth % 2 == 1);
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

bool GameGenerator::SearcherForGen::mateSearchForEvader(Position& pos, int32_t depth) {
    assert(depth % 2 == 0);
    if (depth == 0) {
        return pos.generateAllMoves().empty();
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

    return true;
}

#endif