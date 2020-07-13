#ifndef MIACIS_MCTS_NET_HPP
#define MIACIS_MCTS_NET_HPP

#include "hash_table_for_mcts_net.hpp"
#include "mcts_net_nn.hpp"
#include "../search_options.hpp"

//MCTSを行うクラス
//想定の使い方は局面を放り投げて探索せよと投げることか
//なのでSearcherForPlayと置き換えられるように作れば良さそう
class MCTSNet {
public:
    explicit MCTSNet(const SearchOptions& search_options);

    //探索を行って一番良い指し手を返す関数
    Move think(Position& root, int64_t time_limit);

private:
    //探索に関するオプション
    const SearchOptions& search_options_;

    //置換表は1個
    HashTableForMCTSNet hash_table_;

    //使用するニューラルネットワーク
    NeuralNetworks neural_networks_;
};

#endif //MIACIS_MCTS_NET_HPP