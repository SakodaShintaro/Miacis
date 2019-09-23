#ifndef MIACIS_SEARCHER_USING_SIM_NET_HPP
#define MIACIS_SEARCHER_USING_SIM_NET_HPP

#include"searcher.hpp"
#include"neural_network.hpp"
#include"usi_options.hpp"

struct SimHashEntry {
    int32_t sum_N;
    std::vector<Move> moves;
    std::vector<int32_t> N;
    std::vector<FloatType> nn_policy;
    std::vector<FloatType> state_representation;
    ValueType value;
    bool evaled;

#ifdef USE_CATEGORICAL
    SimHashEntry() :
        sum_N(0), value({}), evaled(false) {}
#else
    SimHashEntry() :
        sum_N(0), value(0.0), evaled(false){}
#endif
};

class SearcherUsingSimNet {
public:
    SearcherUsingSimNet(const UsiOptions& usi_options, NeuralNetwork evaluator) :
        usi_options_(usi_options), evaluator_(std::move(evaluator)) {}

    //探索を行って一番良い指し手を返す関数
    Move think(Position& root, int64_t random_turn);

    //探索を行って一番良い指し手を返す関数
    Move thinkMCTS(Position& root, int64_t random_turn);

private:
    //--------------------------
    //    共通する変数,関数など
    //--------------------------
    //node局面におけるi番目の指し手の行動価値を返す関数
    ValueType QfromNextValue(std::vector<Move> moves, int32_t i) const;

    //選択
    Move select(const std::vector<Move>& moves);

    //展開
    void expand(const Position& pos, const std::vector<Move>& moves,
                const std::vector<FloatType>& state_rep);

    //置換表
    std::map<std::vector<Move>, SimHashEntry> hash_table_;

    //USIOption
    const UsiOptions& usi_options_;

    //局面評価に用いるネットワーク
    NeuralNetwork evaluator_;
};

#endif //MIACIS_SEARCHER_FOR_PLAY_HPP