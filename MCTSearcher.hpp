#pragma once
#ifndef MCTSEARCHER_HPP
#define MCTSEARCHER_HPP

#include"types.hpp"
#include"uct_hash_table.hpp"
#include"network.hpp"
#include<vector>
#include<chrono>
#include<atomic>

#ifdef USE_NN
#define USE_MCTS
#endif

class MCTSearcher {
public:
    //コンストラクタ
    MCTSearcher(int64_t hash_size, int64_t thread_num) : hash_table_(hash_size) {}
    
    //一番良い指し手と学習データを返す関数
    std::pair<Move, TeacherType> think(Position& pos);

    static int64_t limit_msec;
    static std::atomic<bool> stop_signal;
    static bool print_usi_info;
    static bool train_mode;

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
};

#endif // !MCTSEARCHER_HPP