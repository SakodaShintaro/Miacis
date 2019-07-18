#ifndef MIACIS_SEARCHER_HPP
#define MIACIS_SEARCHER_HPP

#include"uct_hash_table.hpp"
#include<chrono>

class Searcher {
public:
    static bool stop_signal;

protected:
    explicit Searcher(int64_t hash_size, double C_PUCT) : hash_table_(hash_size),
                                                          C_PUCT_(C_PUCT),
                                                          root_index_(UctHashTable::NOT_EXPANDED),
                                                          time_limit_(LLONG_MAX),
                                                          node_limit_(LLONG_MAX) {}

    //時間制限含め探索を続けるかどうかを判定する関数
    bool shouldStop();

    //今のノードから遷移するべきノードを選択する関数
    int32_t selectMaxUcbChild(const UctHashEntry& node);

    //node局面におけるi番目の指し手の行動価値を返す関数
    ValueType QfromNextValue(const UctHashEntry& node, int32_t i) const;

    //詰み探索
    void mateSearch(Position pos, int32_t depth_limit);
    bool mateSearchForAttacker(Position& pos, int32_t depth);
    bool mateSearchForEvader(Position& pos, int32_t depth);

    //置換表
    UctHashTable hash_table_;

    //C_PUCT
    const double C_PUCT_;

    //時間
    std::chrono::steady_clock::time_point start_;

    //ルート局面のインデックス
    Index root_index_;

    //時間制限(msec),ノード数制限
    int64_t time_limit_;
    int64_t node_limit_;
};

#endif //MIACIS_SEARCHER_HPP