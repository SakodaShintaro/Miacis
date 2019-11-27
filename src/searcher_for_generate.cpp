#include "searcher_for_generate.hpp"
#include "include_switch.hpp"

bool SearcherForGenerate::prepareForCurrPos(Position& root) {
    //古いハッシュを削除
    hash_table_.deleteOldHash(root, false);

    //ルートノードの展開
    std::stack<int32_t> indices;
    std::stack<int32_t> actions;
    root_index_ = expand(root, indices, actions);

    if (root.turnNumber() >= 50) {
        //5手詰めまで探索
        mateSearch(root, 5);
    }

    //合法手が0かどうかを判定して返す
    return !hash_table_[root_index_].moves.empty();
}

void SearcherForGenerate::select(Position& pos) {
    if (hash_table_[root_index_].sum_N == 0) {
        //初回の探索をする前にノイズを加える
        //Alpha Zeroの論文と同じディリクレノイズ
        UctHashEntry& root_node = hash_table_[root_index_];
        constexpr double epsilon = 0.25;
        std::vector<double> dirichlet = dirichletDistribution(root_node.moves.size(), 0.15);
        for (uint64_t i = 0; i < root_node.moves.size(); i++) {
            root_node.nn_policy[i] = (FloatType) ((1.0 - epsilon) * root_node.nn_policy[i] + epsilon * dirichlet[i]);
        }

        root_raw_value_ = root_node.value;
    }

    std::stack<Index> curr_indices;
    std::stack<int32_t> curr_actions;

    Index index = root_index_;
    //ルートでは合法手が一つはあるはず
    assert(!hash_table_[index].moves.empty());

    //未展開の局面に至るまで遷移を繰り返す
    while(index != UctHashTable::NOT_EXPANDED) {
        if (hash_table_[index].moves.empty()) {
            //詰みの場合抜ける
            break;
        }

//        Score repeat_score;
//        if (index != root_index_ && pos.isRepeating(repeat_score)) {
//            //繰り返しが発生している場合も抜ける
//            break;
//        }

        //状態を記録
        curr_indices.push(index);

        //選択
        int32_t action = selectMaxUcbChild(hash_table_[index]);

        //取った行動を記録
        curr_actions.push(action);

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

    //葉の直前ノードを更新
    hash_table_[index].child_indices[action] = leaf_index;

    //バックアップはGPU計算後にやるので局面だけ戻す
    for (uint64_t i = 0; i < move_num; i++) {
        pos.undo();
    }
}

Index SearcherForGenerate::expand(Position& pos, std::stack<int32_t>& indices, std::stack<int32_t>& actions) {
    uint64_t index = hash_table_.findSameHashIndex(pos);

    // 合流先が検知できればそれを返す
    if (index != hash_table_.size()) {
        //GPUに送らないのでこのタイミングでバックアップを行う
        indices.push(index);
        backup(indices, actions);
        return index;
    }

    // 空のインデックスを探す
    index = hash_table_.searchEmptyIndex(pos);

    UctHashEntry& curr_node = hash_table_[index];

    // 現在のノードの初期化
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
#ifdef USE_CATEGORICAL
    curr_node.value = std::array<float, BIN_SIZE>{};
#else
    curr_node.value = 0.0;
#endif

    // ノードを評価
    float score;
    if (pos.isFinish(score)) {
#ifdef USE_CATEGORICAL
        curr_node.value = onehotDist(score);
#else
        curr_node.value = score;
#endif
        curr_node.evaled = true;
        //GPUに送らないのでこのタイミングでバックアップを行う
        indices.push(index);
        backup(indices, actions);
    } else {
        //特徴量の追加
        std::vector<FloatType> this_feature = pos.makeFeature();
        input_queue_.insert(input_queue_.end(), this_feature.begin(), this_feature.end());

        //インデックス,行動の履歴およびidを追加
        indices.push(index);
        index_queue_.push_back(indices);
        action_queue_.push_back(actions);
        id_queue_.push_back(id_);
    }

    return index;
}

void SearcherForGenerate::backup(std::stack<int32_t>& indices, std::stack<int32_t>& actions) {
    assert(indices.size() == actions.size() + 1);
    int32_t leaf = indices.top();
    indices.pop();
    ValueType value = hash_table_[leaf].value;
    static constexpr FloatType LAMBDA = 1.0;

    //バックアップ
    while (!actions.empty()) {
        int32_t index = indices.top();
        indices.pop();

        int32_t action = actions.top();
        actions.pop();

        //手番が変わるので反転
#ifdef USE_CATEGORICAL
        std::reverse(value.begin(), value.end());
#else
        value = MAX_SCORE + MIN_SCORE - value;
#endif

        // 探索結果の反映
        hash_table_[index].N[action]++;
        hash_table_[index].sum_N++;

        ValueType curr_v = hash_table_[index].value;
        float alpha = 1.0f / (hash_table_[index].sum_N + 1);
        hash_table_[index].value += alpha * (value - curr_v);
        value = LAMBDA * value + (1.0f - LAMBDA) * curr_v;

        assert(!hash_table_[index].moves.empty());
    }
}

OneTurnElement SearcherForGenerate::resultForCurrPos(Position& root) {
    const UctHashEntry& root_node = hash_table_[root_index_];
    assert(!root_node.moves.empty());
    assert(root_node.sum_N != 0);
    const std::vector<int32_t>& N = root_node.N;

    //探索回数最大の手を選択する
    int32_t best_index = (int32_t)(std::max_element(N.begin(), N.end()) - N.begin());

    //選択した着手の勝率の算出
    //詰みのときは未展開であることに注意する
#ifdef USE_CATEGORICAL
    FloatType best_value = (root_node.child_indices[best_index] == UctHashTable::NOT_EXPANDED ? MAX_SCORE :
                    expOfValueDist(QfromNextValue(root_node, best_index)));
#else
    FloatType best_value = (root_node.child_indices[best_index] == UctHashTable::NOT_EXPANDED ? MAX_SCORE :
                    QfromNextValue(root_node, best_index));
#endif

    //教師データを作成
    OneTurnElement element;

    //valueのセット
#ifdef USE_CATEGORICAL
    element.value_teacher = valueToIndex(best_value);
#else
    element.value_teacher = (FloatType)best_value;
#endif

    //policyのセット
    assert(root_node.sum_N == std::accumulate(N.begin(), N.end(), 0));

    if (root.turnNumber() < usi_options_.random_turn) {
        //分布に従ってランダムに行動選択
        //探索回数を正規化した分布
        //探索回数のsoftmaxを取ることを検討したほうが良いかもしれない
        std::vector<FloatType> N_dist(root_node.moves.size());
        //行動価値のsoftmaxを取った分布
        std::vector<FloatType> Q_dist(root_node.moves.size());
        assert(root_node.sum_N == std::accumulate(N.begin(), N.end(), 0));
        for (uint64_t i = 0; i < root_node.moves.size(); i++) {
            assert(0 <= N[i] && N[i] <= root_node.sum_N);

            //探索回数を正規化
            N_dist[i] = (FloatType)N[i] / root_node.sum_N;

            //選択回数が0ならMIN_SCORE
            //選択回数が0ではないのに未展開なら詰み探索が詰みを発見したということなのでMAX_SCORE
            //その他は普通に計算
#ifdef USE_CATEGORICAL
            Q_dist[i] = (N[i] == 0 ? MIN_SCORE : root_node.child_indices[i] == UctHashTable::NOT_EXPANDED ? MAX_SCORE : expOfValueDist(QfromNextValue(root_node, i)));
#else
            Q_dist[i] = (N[i] == 0 ? MIN_SCORE : root_node.child_indices[i] == UctHashTable::NOT_EXPANDED ? MAX_SCORE : QfromNextValue(root_node, i));
#endif
        }
        Q_dist = softmax(Q_dist, usi_options_.temperature_x1000 / 1000.0f);

        //教師分布のセット
        //(1)どちらの分布を使うべきか
        //(2)実際に行動選択をする分布と一致しているべきか
        //など要検討
        for (uint64_t i = 0; i < root_node.moves.size(); i++) {
            //N_distにQ_distの値を混ぜ込む
            N_dist[i] = (1 - Q_dist_lambda_) * N_dist[i] + Q_dist_lambda_ * Q_dist[i];

            //N_distを教師分布とする
            element.policy_teacher.push_back({ root_node.moves[i].toLabel(), N_dist[i] });
        }

        //N_distに従って行動選択
        element.move = root_node.moves[randomChoose(N_dist)];
    } else {
        //最良の行動を選択
        element.policy_teacher.push_back({ root_node.moves[best_index].toLabel(), 1.0f });
        element.move = root_node.moves[best_index];
    }

    element.score = best_value;

    //priorityを計算する用にNNの出力をセットする
    element.nn_output_policy.resize(POLICY_DIM, 0.0);
    for (uint64_t i = 0; i < root_node.moves.size(); i++) {
        element.nn_output_policy[root_node.moves[i].toLabel()] = root_node.nn_policy[i];
    }
    element.nn_output_value = root_raw_value_;

    return element;
}

std::vector<double> SearcherForGenerate::dirichletDistribution(uint64_t k, double alpha) {
    static std::random_device seed;
    static std::default_random_engine engine(seed());
    static constexpr double eps = 0.000000001;
    std::gamma_distribution<double> gamma(alpha, 1.0);
    std::vector<double> dirichlet(k);
    double sum = 0.0;
    for (uint64_t i = 0; i < k; i++) {
        sum += (dirichlet[i] = std::max(gamma(engine), eps));
    }
    for (uint64_t i = 0; i < k; i++) {
        dirichlet[i] /= sum;
    }
    return dirichlet;
}