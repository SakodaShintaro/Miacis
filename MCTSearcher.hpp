#ifndef MCTSEARCHER_HPP
#define MCTSEARCHER_HPP

#include"types.hpp"
#include"uct_hash_table.hpp"
#include"neural_network.hpp"
#include"usi_options.hpp"
#include<vector>
#include<chrono>
#include<atomic>

template <class Var>
class MCTSearcher {
public:
    //コンストラクタ
    MCTSearcher(int64_t hash_size, int64_t thread_num, NeuralNetwork<Var>& nn) : hash_table_(hash_size), evaluator_(nn) {}
    
    //一番良い指し手と学習データを返す関数
    std::pair<Move, TeacherType> think(Position& pos);

private:
    //再帰する探索関数
#ifdef USE_CATEGORICAL
    std::array<CalcType, BIN_SIZE> uctSearch(Position& pos, Index current_index);
#else
    CalcType uctSearch(Position& pos, Index current_index);
#endif

    //ノードを展開する関数
    Index expandNode(Position& pos);

    //ノードを評価する関数
    void evalNode(Position& pos, Index index);

    //経過時間が持ち時間をオーバーしていないか確認する関数
    bool isTimeOver();

    //時間経過含め、playoutの回数なども考慮しplayoutを続けるかどうかを判定する関数
    bool shouldStop();

    //PVを取得する関数
    std::vector<Move> getPV() const;

    //情報をUSIプロトコルに従って標準出力に出す関数
    void printUSIInfo() const;

    //Ucbを計算して最大値を持つインデックスを返す
#ifdef USE_CATEGORICAL
    static int32_t selectMaxUcbChild(const UctHashEntry& current_node, double curr_best_winrate);
#else
    static int32_t selectMaxUcbChild(const UctHashEntry& current_node);
#endif

    //ディリクレ分布に従ったものを返す関数
    static std::vector<double> dirichletDistribution(int32_t k, double alpha);

    //置換表
    UctHashTable hash_table_;

    //Playout回数
    uint32_t playout_num;

    Index current_root_index_;

    //時間
    std::chrono::steady_clock::time_point start_;

    //局面評価に用いるネットワーク
    NeuralNetwork<Var>& evaluator_;
};

template <class Var>
std::pair<Move, TeacherType> MCTSearcher<Var>::think(Position& root) {
    //思考開始時間をセット
    start_ = std::chrono::steady_clock::now();

    //古いハッシュを削除
    hash_table_.deleteOldHash(root, false);

    //ルートノードの展開
    current_root_index_ = expandNode(root);
    auto& current_node = hash_table_[current_root_index_];

    //合法手が0だったら投了
    if (current_node.child_num == 0) {
        return { NULL_MOVE, TeacherType() };
    }

    //ノイズを加える
    //Alpha Zeroの論文と同じディリクレノイズ
    if (usi_option.train_mode) {
        constexpr double epsilon = 0.25;
        auto dirichlet = dirichletDistribution(current_node.child_num, 0.15);
        for (int32_t i = 0; i < current_node.child_num; i++) {
            current_node.nn_rates[i] = (CalcType)((1.0 - epsilon) * current_node.nn_rates[i] + epsilon * dirichlet[i]);
        }
    }

    //初期化
    playout_num = 0;

    //プレイアウトを繰り返す
    //探索回数が閾値を超える、または探索が打ち切られたらループを抜ける
    while (playout_num < usi_option.playout_limit) {
        //探索回数を1回増やす
        playout_num++;

        //1回プレイアウト
        uctSearch(root, current_root_index_);

        //探索を打ち切るか確認
        if (shouldStop() || !hash_table_.hasEnoughSize()) {
            break;
        }
    }

    const auto& child_move_counts = current_node.child_move_counts;

    if (usi_option.print_usi_info) {
        printUSIInfo();
        root.print();
        auto root_moves = current_node.legal_moves;
        for (int32_t i = 0; i < current_node.child_num; i++) {
#ifdef USE_CATEGORICAL
            double v = 0.0;
            if (child_move_counts[i] != 0) {
                for (int32_t j = 0; j < BIN_SIZE; j++) {
                    v += VALUE_WIDTH * (0.5 + j) * (current_node.child_wins[i][j] / child_move_counts[i]);
                }
            }
            printf("%3d: move_count = %6d, nn_rate = %.5f, win_rate = %7.5f, ", i, child_move_counts[i],
                current_node.nn_rates[i], v);
            root_moves[i].print();
#else
            printf("%3d: move_count = %6d, nn_rate = %.5f, win_rate = %7.5f, ", i, child_move_counts[i],
                   current_node.nn_rates[i], (child_move_counts[i] > 0 ? current_node.child_wins[i] / child_move_counts[i] : 0));
            root_moves[i].print();
#endif
        }
    }

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
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        teacher[POLICY_DIM + i] = current_node.child_wins[best_index][i] / child_move_counts[best_index];
    }
#else
    teacher.value = (CalcType)best_wp;
#endif

    //訪問回数に基づいた分布を得る
    std::vector<CalcType> distribution(current_node.child_num);
    for (int32_t i = 0; i < current_node.child_num; i++) {
        distribution[i] = (CalcType)child_move_counts[i] / current_node.move_count;
        assert(0.0 <= distribution[i] && distribution[i] <= 1.0);
    }

    //最善手
    Move best_move = (root.turn_number() < usi_option.random_turn ?
                      current_node.legal_moves[randomChoise(distribution)] :
                      current_node.legal_moves[best_index]);

    best_move.score = (Score)(best_wp);
    teacher.policy = best_move.toLabel();

    return { best_move, teacher };
}

template <class Var>
ValueType MCTSearcher<Var>::uctSearch(Position & pos, Index current_index) {
    auto& current_node = hash_table_[current_index];

    if (current_node.child_num == 0) {
#ifdef USE_CATEGORICAL
        std::array<CalcType, BIN_SIZE> lose_value_dist;
        for (int32_t i = 0; i < BIN_SIZE; i++) {
            lose_value_dist[i] = (i == 0 ? 1.0f : 0.0f);
        }
        return lose_value_dist;
#else
        return 0.0;
#endif
    }

    auto& child_indices = current_node.child_indices;

    // UCB値が最大の手を求める
#ifdef USE_CATEGORICAL
    const auto& child_move_counts = current_node.child_move_counts;
    int32_t selected_index = (int32_t)(std::max_element(child_move_counts.begin(), child_move_counts.end()) - child_move_counts.begin());
    double best_wp = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        double v = (child_move_counts[selected_index] == 0 ? 0.0 :
            current_node.child_wins[selected_index][i] / child_move_counts[selected_index]);
        best_wp += VALUE_WIDTH * (0.5 + i) * v;
    }
    auto next_index = selectMaxUcbChild(current_node, best_wp);
#else
    auto next_index = selectMaxUcbChild(current_node);
#endif

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
        result = hash_table_[index].value;
    } else {
        // 手番を入れ替えて1手深く読む
        result = uctSearch(pos, child_indices[next_index]);
    }
    //手番が変わっているので反転
    result = reverse(result);

    // 探索結果の反映
    current_node.win_sum += result;
    current_node.move_count++;
    current_node.child_wins[next_index] += result;
    current_node.child_move_counts[next_index]++;

    // 手を戻す
    pos.undo();

    return result;
}

template <class Var>
Index MCTSearcher<Var>::expandNode(Position& pos) {
    auto index = hash_table_.findSameHashIndex(pos.hash_value(), pos.turn_number());

    // 合流先が検知できればそれを返す
    if (index != hash_table_.size()) {
        return index;
    }

    // 空のインデックスを探す
    index = hash_table_.searchEmptyIndex(pos.hash_value(), pos.turn_number());

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
        current_node.value_dist[i] = 0.0;
        current_node.win_sum[i] = 0.0;
        for (int32_t j = 0; j < current_node.child_num; j++) {
            current_node.child_wins[j][i] = 0.0;
        }
    }
#else
    current_node.value = 0.0;
    current_node.win_sum = 0.0;
    current_node.child_wins = std::vector<float>(current_node.child_num, 0.0);
#endif

    // ノードを評価
    if (current_node.child_num > 0) {
        evalNode(pos, index);
    } else {
        if (pos.lastMove().isDrop() && (kind(pos.lastMove().subject()) == PAWN)) {
            //打ち歩詰め
#ifdef USE_CATEGORICAL
            for (int32_t i = 0; i < BIN_SIZE; i++) {
                current_node.value_dist[i] = (i == BIN_SIZE - 1 ? 1.0f : 0.0f);
            }
#else
            current_node.value = 1.0;
#endif
        } else {
            //詰み
#ifdef USE_CATEGORICAL
            for (int32_t i = 0; i < BIN_SIZE; i++) {
                current_node.value_dist[i] = (i == 0 ? 1.0f : 0.0f);
            }
#else
            current_node.value = -1.0f;
#endif
        }
        current_node.evaled = true;
    }

    return index;
}

template <class Var>
void MCTSearcher<Var>::evalNode(Position& pos, Index index) {
    auto& current_node = hash_table_[index];
    std::vector<float> legal_move_policy(current_node.child_num);

    auto policy_and_value = evaluator_.policyAndValue(pos);

    for (int32_t i = 0; i < current_node.child_num; i++) {
        legal_move_policy[i] = policy_and_value.first[current_node.legal_moves[i].toLabel()];
    }

    //ノードの値を計算
    current_node.value = policy_and_value.second;

    //softmax分布にする
    current_node.nn_rates = softmax(legal_move_policy);

    Score repeat_score;
    if (pos.isRepeating(repeat_score)) {
#ifdef USE_CATEGORICAL
        current_node.value_dist = onehotDist(sigmoid(repeat_score, CP_GAIN));
#else
        current_node.value = repeat_score;
#endif
    }

    current_node.evaled = true;
}

template <class Var>
bool MCTSearcher<Var>::isTimeOver() {
    //時間のチェック
    auto now_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now_time - start_);
    return (elapsed.count() >= usi_option.limit_msec - usi_option.byoyomi_margin);
}

template <class Var>
bool MCTSearcher<Var>::shouldStop() {
    if (isTimeOver()) {
        return true;
    }
    return false;

//    // 探索回数が最も多い手と次に多い手を求める
//    int32_t max1 = 0, max2 = 0;
//    for (auto e : hash_table_[current_root_index_].child_move_counts) {
//        if (e > max1) {
//            max2 = max1;
//            max1 = e;
//        } else if (e > max2) {
//            max2 = e;
//        }
//    }
//
//    // 残りの探索を全て次善手に費やしても最善手を超えられない場合は探索を打ち切る
//    return (max1 - max2) > (usi_option.playout_limit - playout_num);
}

template <class Var>
std::vector<Move> MCTSearcher<Var>::getPV() const {
    std::vector<Move> pv;
    for (Index curr_node_index = current_root_index_; curr_node_index != UctHashTable::NOT_EXPANDED && hash_table_[curr_node_index].child_num != 0; ) {
        const auto& child_move_counts = hash_table_[curr_node_index].child_move_counts;
        Index next_index = (int32_t)(std::max_element(child_move_counts.begin(), child_move_counts.end()) - child_move_counts.begin());
        pv.push_back(hash_table_[curr_node_index].legal_moves[next_index]);
        curr_node_index = hash_table_[curr_node_index].child_indices[next_index];
    }

    return pv;
}

template <class Var>
void MCTSearcher<Var>::printUSIInfo() const {
    const auto& current_node = hash_table_[current_root_index_];

    //探索にかかった時間を求める
    auto finish_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_);

    const auto& child_move_counts = current_node.child_move_counts;
    auto selected_index = (int32_t)(std::max_element(child_move_counts.begin(), child_move_counts.end()) - child_move_counts.begin());

    //選択した着手の勝率の算出
#ifdef USE_CATEGORICAL
    double best_wp = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        double v = current_node.child_wins[selected_index][i] / child_move_counts[selected_index];
        best_wp += VALUE_WIDTH * (0.5 + i) * v;
    }
#else
    auto best_wp = (child_move_counts[selected_index] == 0 ? 0.0
                                                           : current_node.child_wins[selected_index] / child_move_counts[selected_index]);
#endif

    //勝率を評価値に変換
    //int32_t cp = inv_sigmoid(best_wp, CP_GAIN);
    int32_t cp = best_wp * 1000;

    printf("info nps %d time %d nodes %d hashfull %d score cp %d pv ",
           (int)(current_node.move_count * 1000 / std::max((long long)elapsed.count(), 1LL)),
           (int)(elapsed.count()),
           current_node.move_count,
           (int)(hash_table_.getUsageRate() * 1000),
           cp);

    auto pv = getPV();
    for (auto m : pv) {
        std::cout << m << " ";
    }
    std::cout << std::endl;
}

template <class Var>
std::vector<double> MCTSearcher<Var>::dirichletDistribution(int32_t k, double alpha) {
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

template <class Var>
int32_t MCTSearcher<Var>::selectMaxUcbChild(const UctHashEntry & current_node) {
#ifdef USE_CATEGORICAL
    const auto& child_move_counts = current_node.child_move_counts;
    int32_t selected_index = (int32_t)(std::max_element(child_move_counts.begin(), child_move_counts.end()) - child_move_counts.begin());
    double best_wp = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        double v = (child_move_counts[selected_index] == 0 ? 0.0 :
                    current_node.child_wins[selected_index][i] / child_move_counts[selected_index]);
        best_wp += VALUE_WIDTH * (0.5 + i) * v;
    }
#endif

    const auto& child_move_counts = current_node.child_move_counts;

    // ucb = Q(s, a) + U(s, a)
    // Q(s, a) = W(s, a) / N(s, a)
    // U(s, a) = C_PUCT * P(s, a) * sqrt(sum_b(B(s, b)) / (1 + N(s, a))
    constexpr double C_PUCT = 1.0;

    int32_t max_index = -1;
    double max_value = -1;
    for (int32_t i = 0; i < current_node.child_num; i++) {
#ifdef USE_CATEGORICAL
        double Q;
        if (child_move_counts[i] == 0) {
            Q = 0.5;
        } else {
            Q = 0.0;
            for (int32_t j = std::min((int32_t)(curr_best_winrate * BIN_SIZE) + 1, BIN_SIZE - 1); j < BIN_SIZE; j++) {
                Q += current_node.child_wins[i][j] / child_move_counts[i];
            }
        }
#else
        double Q = (child_move_counts[i] == 0 ? 0.0 : current_node.child_wins[i] / child_move_counts[i]);
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

#endif // !MCTSEARCHER_HPP