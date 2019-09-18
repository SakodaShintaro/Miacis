#include "searcher_using_sim_net.hpp"

Move SearcherUsingSimNet::think(Position &root, int64_t random_turn) {
    std::vector<FloatType> root_features = root.makeFeature();
    std::vector<FloatType> input_features;

    std::vector<Move> moves = root.generateAllMoves();

    //バッチ数分貯める
    for (uint64_t i = 0; i < moves.size(); i++) {
        input_features.insert(input_features.end(), root_features.begin(), root_features.end());
    }

    torch::Tensor root_state_representation = evaluator_->encodeStates(input_features);

    torch::Tensor move_representations = evaluator_->encodeActions(moves);

    torch::Tensor predicted_next_state_representations = evaluator_->predictTransition(root_state_representation, move_representations);

    torch::Tensor values = evaluator_->decodeValue(predicted_next_state_representations).cpu();

    std::vector<FloatType> values_vec(values.data<FloatType>(), values.data<FloatType>() + moves.size());

    FloatType best_value = MIN_SCORE - 1;
    uint64_t best_index = 0;
    for (uint64_t i = 0; i < moves.size(); i++) {
        if (values_vec[i] > best_value) {
            best_value = values_vec[i];
            best_index = i;
        }
    }

    return moves[best_index];
}