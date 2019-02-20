#ifndef MIACIS_SEARCHER_HPP
#define MIACIS_SEARCHER_HPP

#include"uct_hash_table.hpp"
#include<chrono>
#include<atomic>

class Searcher {
public:
protected:
    explicit Searcher(int64_t hash_size) : hash_table_(hash_size) {}

    //経過時間が持ち時間をオーバーしていないか確認する関数
    bool isTimeOver();

    //時間経過含め、playoutの回数なども考慮しplayoutを続けるかどうかを判定する関数
    bool shouldStop();

    //今のノードから遷移するべきノードを選択する関数
    static int32_t selectMaxUcbChild(const UctHashEntry& current_node);

    //詰み探索
    bool mateSearchForAttacker(Position& pos, int32_t depth);
    bool mateSearchForEvader(Position& pos, int32_t depth);

    //置換表
    UctHashTable hash_table_;

    //時間
    std::chrono::steady_clock::time_point start_;

    //ルート局面のインデックス
    Index current_root_index_;
};

#endif //MIACIS_SEARCHER_HPP