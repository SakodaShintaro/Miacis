#include "searcher_using_sim_net.hpp"

Move SearcherUsingSimNet::think(Position &root, int64_t random_turn) {
    root.print();

    //現局面の合法手
    std::vector<Move> moves = root.generateAllMoves();

    //合法手をそれぞれ表現としたもの
    torch::Tensor move_representations = evaluator_->encodeActions(moves);

    //現局面の特徴量
    std::vector<FloatType> root_features = root.makeFeature();

    //上の現局面の特徴量を合法手の数だけ複製する
    std::vector<FloatType> input_features;
    for (uint64_t i = 0; i < moves.size(); i++) {
        input_features.insert(input_features.end(), root_features.begin(), root_features.end());
    }

    //現局面の合法手分の表現
    torch::Tensor root_state_representations = evaluator_->encodeStates(input_features);

    //予測した次状態表現
    torch::Tensor predicted_next_state_representations = evaluator_->predictTransition(root_state_representations, move_representations);

    //真の次状態の特徴を取得
    std::vector<FloatType> next_features;
    for (const Move& move : moves) {
        root.doMove(move);
        std::vector<FloatType> next_feature = root.makeFeature();
        next_features.insert(next_features.end(), next_feature.begin(), next_feature.end());
        root.undo();
    }

    //真の次状態表現を取得
    torch::Tensor next_state_representations = evaluator_->encodeStates(next_features);

    //損失を計算
    torch::Tensor transition_loss = evaluator_->transitionLoss(predicted_next_state_representations,
                                                                         next_state_representations);
//    std::cout <<  transition_loss << std::endl;

    //推論した価値を予測
    torch::Tensor values = evaluator_->decodeValue(predicted_next_state_representations).cpu();
    std::vector<FloatType> values_vec(values.data<FloatType>(), values.data<FloatType>() + moves.size());

    //現局面について方策を推論
    torch::Tensor root_representation = evaluator_->encodeStates(root_features);
    torch::Tensor policy = evaluator_->decodePolicy(root_representation)[0];
    std::vector<FloatType> policy_vec;
    for (const Move& move : moves) {
        policy_vec.push_back(policy[move.toLabel()].item<FloatType>());
    }
    policy_vec = softmax(policy_vec);

    //auto policy_and_value = evaluator_->policyAndValueBatch(root_features);
    //policy_vec = policy_and_value.first[0];
    //policy_vec = softmax(policy_vec);

    //真の状態表現から推論する価値
    torch::Tensor true_values = evaluator_->decodeValue(next_state_representations).cpu();
    std::vector<FloatType> true_values_vec(true_values.data<FloatType>(), true_values.data<FloatType>() + moves.size());

    FloatType best_value = MIN_SCORE - 1;
    uint64_t best_index = 0;
    for (uint64_t i = 0; i < moves.size(); i++) {
//        std::cout << std::showpos << std::fixed << std::setprecision(4) << -values_vec[i] << " ";
//        std::cout << std::showpos << std::fixed << std::setprecision(4) << -true_values_vec[i] << " ";
//        std::cout << std::showpos << std::fixed << std::setprecision(4) << policy_vec[i] << " ";
//        std::cout << std::noshowpos;
//        moves[i].print();
//        std::cout << std::endl;
        if (-true_values_vec[i] > best_value) {
            best_value = -true_values_vec[i];
            best_index = i;
        }
    }

    std::cout << "info score cp " << (int64_t)(best_value * 1000) << std::endl;

    return moves[best_index];
}

Move SearcherUsingSimNet::thinkMCTS(Position& root, int64_t random_turn) {
    //前回の探索で使っている可能性があるので置換表をクリア
    hash_table_.clear();

    //ルート局面の状態を設定する
    torch::Tensor root_rep_tensor = evaluator_->encodeStates(root.makeFeature()).cpu();
    std::vector<FloatType> root_rep(root_rep_tensor.data<FloatType>(), root_rep_tensor.data<FloatType>() + 9 * 9 * 64);
    expand(root, std::vector<Move>(), root_rep, true);

    //合法手が0だったら投了
    if (hash_table_[std::vector<Move>()].moves.empty()) {
        return NULL_MOVE;
    }

    //規定回数まで選択
    for (int64_t i = 0; i < usi_options_.search_limit; i++) {
        Position pos = root;
        std::vector<FloatType> state_rep = root_rep;
        std::vector<Move> moves;

        //選択
        while (hash_table_.count(moves)) {
            //保存しておいた状態表現を取得
            state_rep = hash_table_[moves].state_representation;

            if (hash_table_[moves].moves.empty()) {
                //詰みの場合抜ける
                break;
            }

            if (pos.turnNumber() > usi_options_.draw_turn) {
                //手数が制限まで達している場合も抜ける
                break;
            }

            Score repeat_score;
            if (!moves.empty() && pos.isRepeating(repeat_score)) {
                //繰り返しが発生している場合も抜ける
                break;
            }

            if (!moves.empty() && hash_table_[moves].nn_policy.size() != hash_table_[moves].moves.size()) {
                //policyが展開されていなかったら抜ける
                break;
            }

            //行動を選んで遷移
            Move move = select(moves);
            moves.push_back(move);
            pos.doMove(move);
        }

        //状態表現の遷移
        if (!moves.empty()) {
            state_rep = evaluator_->predictTransition(state_rep, moves.back());
        }

        //リーフノードを展開
        expand(pos, moves, state_rep, false);

        ValueType value = hash_table_[moves].value;

        //バックアップ
        while (!moves.empty()) {
            Move last_move = moves.back();
            moves.pop_back();

            //手番が変わるので反転
#ifdef USE_CATEGORICAL
            std::reverse(value.begin(), value.end());
#else
            value = MAX_SCORE + MIN_SCORE - value;
#endif

            SimHashEntry& node = hash_table_[moves];
            node.sum_N++;
            node.N[std::find(node.moves.begin(), node.moves.end(), last_move) - node.moves.begin()]++;
            ValueType curr_v = node.value;
            FloatType alpha = 1.0f / (node.sum_N + 1);
            node.value += alpha * (value - curr_v);
        }
    }

    SimHashEntry& root_node = hash_table_[std::vector<Move>()];

    //ソートするために構造体を準備
    struct MoveWithInfo {
        Move move;
        int32_t N;
        FloatType nn_output_policy, Q, softmaxed_Q;
        bool operator<(const MoveWithInfo& rhs) const {
            return Q < rhs.Q;
        }
        bool operator>(const MoveWithInfo& rhs) const {
            return Q > rhs.Q;
        }
    };

    std::vector<FloatType> Q(root_node.moves.size());
    for (uint64_t i = 0; i < root_node.moves.size(); i++) {
#ifdef USE_CATEGORICAL
        Q[i] = expOfValueDist(QfromNextValue(std::vector<Move>(), i));
#else
        Q[i] = QfromNextValue(std::vector<Move>(), i);
#endif
    }
    std::vector<FloatType> softmaxed_Q = softmax(Q, usi_options_.temperature_x1000 / 1000.0f);

    if (usi_options_.print_policy_num) {
        std::vector<MoveWithInfo> moves_with_info(root_node.moves.size());
        for (uint64_t i = 0; i < root_node.moves.size(); i++) {
            moves_with_info[i].move = root_node.moves[i];
            moves_with_info[i].nn_output_policy = root_node.nn_policy[i];
            moves_with_info[i].N = root_node.N[i];
            moves_with_info[i].Q = Q[i];
            moves_with_info[i].softmaxed_Q = softmaxed_Q[i];
        }

        std::sort(moves_with_info.begin(), moves_with_info.end());

        for (uint64_t i = std::max((int64_t) 0, (int64_t) root_node.moves.size() - usi_options_.print_policy_num);
             i < root_node.moves.size(); i++) {
            printf("info string %03lu  %05.1f  %05.1f  %05.1f  %+0.3f  ", root_node.moves.size() - i,
                   moves_with_info[i].nn_output_policy * 100.0,
                   moves_with_info[i].N * 100.0 / root_node.sum_N,
                   moves_with_info[i].softmaxed_Q * 100,
                   moves_with_info[i].Q);
            moves_with_info[i].move.print();
        }
        std::cout << "info string 順位 NN出力 探索割合 価値分布 価値" << std::endl;
    }

    //評価値の出力
    uint64_t max_index = std::max_element(root_node.N.begin(), root_node.N.end()) - root_node.N.begin();
#ifdef USE_CATEGORICAL
    FloatType best_value = expOfValueDist(QfromNextValue(std::vector<Move>(), max_index));
#else
    FloatType best_value = QfromNextValue(std::vector<Move>(), max_index);
#endif
    std::cout << "info score cp " << (int64_t)(best_value * 1000) << std::endl;

    //行動選択
    if (root.turnNumber() < usi_options_.random_turn) {
        return root_node.moves[randomChoose(softmaxed_Q)];
    } else {
        return root_node.moves[max_index];
    }
}

Move SearcherUsingSimNet::select(const std::vector<Move>& moves) {
    const SimHashEntry& node = hash_table_[moves];
#ifdef USE_CATEGORICAL
    int32_t best_index = std::max_element(node.N.begin(), node.N.end()) - node.N.begin();
    FloatType best_value = expOfValueDist(QfromNextValue(moves, best_index));
#endif

    int32_t max_index = -1;
    FloatType max_value = MIN_SCORE - 1;

    const int32_t sum = node.sum_N;
    for (uint64_t i = 0; i < node.moves.size(); i++) {
        FloatType U = std::sqrt(sum + 1) / (node.N[i] + 1);

#ifdef USE_CATEGORICAL
        FloatType P = 0.0;
        ValueType Q_dist = QfromNextValue(moves, i);
        FloatType Q = expOfValueDist(Q_dist);
        for (int32_t j = std::min(valueToIndex(best_value) + 1, BIN_SIZE - 1); j < BIN_SIZE; j++) {
            P += Q_dist[j];
        }
        FloatType ucb = Q + node.nn_policy[i] * U + P;
#else
        FloatType Q = (node.N[i] == 0 ? MIN_SCORE : QfromNextValue(moves, i));
        FloatType ucb = Q + node.nn_policy[i] * U;
#endif

        if (ucb > max_value) {
            max_value = ucb;
            max_index = i;
        }
    }
    assert(0 <= max_index && max_index < (int32_t)node.moves.size());
    return node.moves[max_index];
}

void SearcherUsingSimNet::expand(const Position& pos, const std::vector<Move>& moves,
                                 const std::vector<FloatType>& state_rep, bool force) {
    SimHashEntry& curr_node = hash_table_[moves];
    curr_node.moves = pos.generateAllMoves();
    curr_node.sum_N = 0;
    curr_node.evaled = true;
    curr_node.N.assign(curr_node.moves.size(), 0);
    curr_node.state_representation = state_rep;

    // ノードを評価
    Score repeat_score;
    if (!force && pos.isRepeating(repeat_score)) {
        //繰り返し
#ifdef USE_CATEGORICAL
        curr_node.value = onehotDist(repeat_score);
#else
        curr_node.value = repeat_score;
#endif
    } else if (!force && pos.turnNumber() > usi_options_.draw_turn) {
        FloatType value = (MAX_SCORE + MIN_SCORE) / 2;
#ifdef USE_CATEGORICAL
        curr_node.value = onehotDist(value);
#else
        curr_node.value = value;
#endif
        curr_node.evaled = true;
    } else if (!force && curr_node.moves.empty()) {
        //打ち歩詰めなら勝ち,そうでないなら負け
        auto v = (pos.isLastMoveDropPawn() ? MAX_SCORE : MIN_SCORE);

#ifdef USE_CATEGORICAL
        curr_node.value = onehotDist(v);
#else
        curr_node.value = v;
#endif
        curr_node.evaled = true;
    } else {
        //GPUで計算
        std::pair<std::vector<PolicyType>, std::vector<ValueType>> p_and_v = evaluator_->decodePolicyAndValueBatch(state_rep);
        curr_node.value = p_and_v.second[0];
        for (const Move& move : curr_node.moves) {
            curr_node.nn_policy.push_back(p_and_v.first[0][move.toLabel()]);
        }
        curr_node.nn_policy = softmax(curr_node.nn_policy);
    }
}

ValueType SearcherUsingSimNet::QfromNextValue(std::vector<Move> moves, int32_t i) const {
#ifdef USE_CATEGORICAL
    moves.push_back(hash_table_.at(moves).moves[i]);
    if (hash_table_.count(moves) == 0) {
        return onehotDist(MIN_SCORE);
    }
    ValueType v = hash_table_.at(moves).value;
    std::reverse(v.begin(), v.end());
    return v;
#else
    moves.push_back(hash_table_.at(moves).moves[i]);
    if (hash_table_.count(moves) == 0) {
        return MIN_SCORE;
    }
    return MAX_SCORE + MIN_SCORE - hash_table_.at(moves).value;
#endif
}
