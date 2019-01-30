#ifndef MIACIS_SEARCHER_FOR_GEN_HPP
#define MIACIS_SEARCHER_FOR_GEN_HPP

#include"types.hpp"
#include"uct_hash_table.hpp"
#include"neural_network.hpp"
#include"usi_options.hpp"
#include"operate_params.hpp"
#include "game_generator.hpp"
#include<vector>
#include<chrono>
#include<thread>
#include<atomic>
#include<stack>
#include<mutex>

class SearcherForGen {
public:
    //コンストラクタ
    SearcherForGen(int64_t hash_size, int32_t id, GameGenerator& gg) : hash_table_(hash_size), id_(id), gg_(gg) {
        lock_node_ = std::vector<std::mutex>(static_cast<unsigned long>(hash_table_.size()));
        clearEvalQueue();
    }

    //一番良い指し手と学習データを返す関数
    std::pair<Move, TeacherType> think(Position& root);

private:
    static constexpr int32_t VIRTUAL_LOSS = 1;

    //再帰する探索関数
    ValueType uctSearch(Position& pos, Index current_index);

    //プレイアウト1回
    void onePlay(Position& pos);

    //ノードを展開する関数
    Index expandNode(Position& pos);

    //時間経過含め、playoutの回数なども考慮しplayoutを続けるかどうかを判定する関数
    bool shouldStop();

    //Ucbを計算して最大値を持つインデックスを返す
    static int32_t selectMaxUcbChild(const UctHashEntry& current_node);

    //ディリクレ分布に従ったものを返す関数
    static std::vector<double> dirichletDistribution(int32_t k, double alpha);

    //評価要求を送る先
    GameGenerator& gg_;

    //このスレッドのid
    int32_t id_;

    //置換表
    UctHashTable hash_table_;

    //Playout回数
    std::atomic<uint32_t> playout_num_;

    //ルート局面のインデックス
    Index current_root_index_;
};

#endif //MIACIS_SEARCHER_FOR_GEN_HPP