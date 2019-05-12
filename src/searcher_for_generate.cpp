#include "searcher_for_generate.hpp"
#include "usi_options.hpp"
#include "operate_params.hpp"

bool SearcherForGenerate::prepareForCurrPos(Position& root) {
    //古いハッシュを削除
    hash_table_.deleteOldHash(root, false);

    //ルートノードの展開
    std::stack<int32_t> indices;
    std::stack<int32_t> actions;
    current_root_index_ = expand(root, indices, actions);

    if (root.turn_number() >= 50) {
        //5手詰めまで探索
        mateSearch(root, 5);
    }

    //合法手が0かどうかを判定して返す
    return !hash_table_[current_root_index_].moves.empty();
}

void SearcherForGenerate::select(Position& pos) {
    if (hash_table_[current_root_index_].sum_N == 0) {
        //初回の探索をする前にノイズを加える
        //Alpha Zeroの論文と同じディリクレノイズ
        auto& root_node = hash_table_[current_root_index_];
        constexpr double epsilon = 0.25;
        auto dirichlet = dirichletDistribution(root_node.moves.size(), 0.15);
        for (int32_t i = 0; i < root_node.moves.size(); i++) {
            root_node.nn_policy[i] = (CalcType) ((1.0 - epsilon) * root_node.nn_policy[i] + epsilon * dirichlet[i]);
        }
    }

    std::stack<Index> curr_indices;
    std::stack<int32_t> curr_actions;

    auto index = current_root_index_;
    //ルートでは合法手が一つはあるはず
    assert(!hash_table_[index].moves.empty());

    //未展開の局面に至るまで遷移を繰り返す
    while(index != UctHashTable::NOT_EXPANDED) {
        if (hash_table_[index].moves.empty()) {
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
        pos.doMove(hash_table_[index].moves[action]);

        //index更新
        index = hash_table_[index].child_indices[action];
    }

    //expandNode内でこれらの情報は壊れる可能性があるので保存しておく
    index = curr_indices.top();
    auto action = curr_actions.top();
    auto move_num = curr_actions.size();

    //今の局面を展開・GPUに評価依頼を投げる
    auto leaf_index = expand(pos, curr_indices, curr_actions);

    //葉の直前ノードを更新
    hash_table_[index].child_indices[action] = leaf_index;

    //バックアップはGPU計算後にやるので局面だけ戻す
    for (int32_t i = 0; i < move_num; i++) {
        pos.undo();
    }
}

Index SearcherForGenerate::expand(Position& pos, std::stack<int32_t>& indices, std::stack<int32_t>& actions) {
    auto index = hash_table_.findSameHashIndex(pos);

    // 合流先が検知できればそれを返す
    if (index != hash_table_.size()) {
        //GPUに送らないのでこのタイミングでバックアップを行う
        indices.push(index);
        backup(indices, actions);
        return index;
    }

    // 空のインデックスを探す
    index = hash_table_.searchEmptyIndex(pos);

    auto& current_node = hash_table_[index];

    // 現在のノードの初期化
    current_node.moves = pos.generateAllMoves();
    current_node.child_indices.assign(current_node.moves.size(), UctHashTable::NOT_EXPANDED);
    current_node.N.assign(current_node.moves.size(), 0);
    current_node.sum_N = 0;
    current_node.evaled = false;
#ifdef USE_CATEGORICAL
    current_node.W.assign(current_node.moves.size(), std::array<float, BIN_SIZE>{});
    current_node.value = std::array<float, BIN_SIZE>{};
#else
    current_node.value = 0.0;
    current_node.W.assign(current_node.moves.size(), 0.0);
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
    if (!current_node.moves.empty()) {
        //特徴量の追加
        auto this_feature = pos.makeFeature();
        input_queue_.insert(input_queue_.end(), this_feature.begin(), this_feature.end());

        //インデックス,行動の履歴およびidを追加
        indices.push(index);
        index_queue_.push_back(indices);
        action_queue_.push_back(actions);
        id_queue_.push_back(id_);
    } else {
        //打ち歩詰めなら勝ち,そうでないなら負け
        auto v = (pos.isLastMoveDropPawn() ? MAX_SCORE : MIN_SCORE);

#ifdef USE_CATEGORICAL
        current_node.value = onehotDist(v);
#else
        current_node.value = v;
#endif
        current_node.evaled = true;
        //GPUに送らないのでこのタイミングでバックアップを行う
        indices.push(index);
        backup(indices, actions);
    }

    return index;
}

void SearcherForGenerate::backup(std::stack<int32_t>& indices, std::stack<int32_t>& actions) {
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
        hash_table_[index].W[action] += value;
        hash_table_[index].N[action]++;
        hash_table_[index].sum_N++;
        assert(!hash_table_[index].moves.empty());
    }
}

std::pair<Move, TeacherType> SearcherForGenerate::resultForCurrPos(Position& root) {
    const auto& current_node = hash_table_[current_root_index_];
    assert(!current_node.moves.empty());
    assert(current_node.sum_N != 0);
    const auto& N = current_node.N;

    //探索回数最大の手を選択する
    int32_t best_index = (int32_t)(std::max_element(N.begin(), N.end()) - N.begin());

    //選択した着手の勝率の算出
#ifdef USE_CATEGORICAL
    auto best_wp = expOfValueDist(current_node.W[best_index]) / N[best_index];
#else
    auto best_wp = (N[best_index] == 0 ? MIN_SCORE : current_node.W[best_index] / N[best_index]);
#endif

    //教師データを作成
    TeacherType teacher;

    //valueのセット
#ifdef USE_CATEGORICAL
    teacher.value = valueToIndex(best_wp);
#else
    teacher.value = (CalcType)best_wp;
#endif

    //policyのセット
    //訪問回数に基づいた分布を得る
    std::vector<CalcType> distribution(current_node.moves.size());
    assert(current_node.sum_N == std::accumulate(N.begin(), N.end(), 0));
    for (int32_t i = 0; i < current_node.moves.size(); i++) {
        distribution[i] = (CalcType)N[i] / current_node.sum_N;
        assert(0 <= N[i] && N[i] <= current_node.sum_N);
    }

    //最善手
    Move best_move = (root.turn_number() < usi_option.random_turn ?
                      current_node.moves[randomChoose(distribution)] :
                      current_node.moves[best_index]);
    best_move.score = (Score)(best_wp);
    teacher.policy = best_move.toLabel();

    return { best_move, teacher };
}

std::vector<double> SearcherForGenerate::dirichletDistribution(uint64_t k, double alpha) {
    static std::random_device seed;
    static std::default_random_engine engine(seed());
    static constexpr double eps = 0.000000001;
    std::gamma_distribution<double> gamma(alpha, 1.0);
    std::vector<double> dirichlet(k);
    double sum = 0.0;
    for (int32_t i = 0; i < k; i++) {
        sum += (dirichlet[i] = std::max(gamma(engine), eps));
    }
    for (int32_t i = 0; i < k; i++) {
        dirichlet[i] /= sum;
    }
    return dirichlet;
}