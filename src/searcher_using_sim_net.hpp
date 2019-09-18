#ifndef MIACIS_SEARCHER_USING_SIM_NET_HPP
#define MIACIS_SEARCHER_USING_SIM_NET_HPP

#include"searcher.hpp"
#include"neural_network.hpp"

class SearcherUsingSimNet {
public:
    SearcherUsingSimNet(NeuralNetwork evaluator) : evaluator_(std::move(evaluator)) {}

    //探索を行って一番良い指し手を返す関数
    Move think(Position& root, int64_t random_turn);

private:
    //--------------------------
    //    共通する変数,関数など
    //--------------------------
    //局面評価に用いるネットワーク
    NeuralNetwork evaluator_;
};

#endif //MIACIS_SEARCHER_FOR_PLAY_HPP