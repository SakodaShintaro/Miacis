#ifndef MIACIS_SEARCHER_HPP
#define MIACIS_SEARCHER_HPP

#include"uct_hash_table.hpp"
#include<chrono>
#include <stack>

class Searcher {
public:
    static bool stop_signal;

protected:
    explicit Searcher(int64_t hash_size, FloatType Q_coeff, FloatType C_PUCT, FloatType P_coeff)
                       : hash_table_(hash_size), Q_coeff_(Q_coeff), C_PUCT_(C_PUCT), P_coeff_(P_coeff),
                         root_index_(UctHashTable::NOT_EXPANDED), time_limit_(LLONG_MAX), node_limit_(LLONG_MAX) {}

    //時間制限含め探索を続けるかどうかを判定する関数
    bool shouldStop();

    //今のノードから遷移するべきノードを選択する関数
    int32_t selectMaxUcbChild(const UctHashEntry& node);

    //node局面におけるi番目の指し手の行動価値を返す関数
    //Scalarのときは実数を一つ、Categoricalのときは分布を返す
    ValueType QfromNextValue(const UctHashEntry& node, int32_t i) const;

    //node局面におけるi番目の指し手の行動価値(期待値)を返す関数
    //Scalarのときは実数をそのまま返し、Categoricalのときはその期待値を返す
    FloatType expQfromNext(const UctHashEntry& node, int32_t i) const;

    //バックアップ
    void backup(std::stack<int32_t>& indices, std::stack<int32_t>& actions);

#ifdef SHOGI
    //詰み探索
    void mateSearch(Position pos, int32_t depth_limit);
    bool mateSearchForAttacker(Position& pos, int32_t depth);
    bool mateSearchForEvader(Position& pos, int32_t depth);
#endif

    //置換表
    UctHashTable hash_table_;

    //Qにかける係数:scalarのときは意味がない気もするがそうでもない？
    //             categoricalのときに使うのがメイン
    const FloatType Q_coeff_;

    //C_PUCT
    const FloatType C_PUCT_;

    //Pにかける係数:scalarのときは使わない
    const FloatType P_coeff_;

    //時間
    std::chrono::steady_clock::time_point start_;

    //ルート局面のインデックス
    Index root_index_;

    //時間制限(msec),ノード数制限
    int64_t time_limit_;
    int64_t node_limit_;
};

#endif //MIACIS_SEARCHER_HPP