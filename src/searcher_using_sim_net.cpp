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
