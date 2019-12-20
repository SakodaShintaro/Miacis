#ifndef MIACIS_SEARCHER_HPP
#define MIACIS_SEARCHER_HPP

#include"uct_hash_table.hpp"
#include"usi_options.hpp"
#include<chrono>
#include<stack>

//GPUに対する評価要求を溜めるキュー
//hash_tableのindexに相当する入力inputを計算して適切な場所に書き込むことを要求する
struct GPUQueue {
    std::vector<float> inputs;
    std::vector<std::reference_wrapper<UctHashTable>> hash_tables;
    std::vector<Index> indices;
};

//評価要求を受けた後バックアップすべく情報を溜めておくキュー
//探索木を降りていった順にノードのindexと行動のindexを保存しておく
struct BackupQueue {
    std::vector<std::stack<Index>> indices;
    std::vector<std::stack<int32_t>> actions;
};

class Searcher {
public:
    static bool stop_signal;

private:
    explicit Searcher(const UsiOptions& usi_options, UctHashTable& hash_table, GPUQueue& gpu_queue)
                       : hash_table_(hash_table), usi_options_(usi_options), root_index_(UctHashTable::NOT_EXPANDED),
                         time_limit_(LLONG_MAX), node_limit_(LLONG_MAX), gpu_queue_(gpu_queue) {}

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

    //再帰しない探索関数
    void select(Position& pos);

    //ノードを展開する関数
    Index expand(Position& pos, std::stack<int32_t>& indices, std::stack<int32_t>& actions);

    //バックアップ
    void backup(std::stack<int32_t>& indices, std::stack<int32_t>& actions);

#ifdef SHOGI
    //詰み探索
    void mateSearch(Position pos, int32_t depth_limit);
    bool mateSearchForAttacker(Position& pos, int32_t depth);
    bool mateSearchForEvader(Position& pos, int32_t depth);
#endif
    //VIRTUAL_LOSSの大きさ
    static constexpr int32_t VIRTUAL_LOSS = 1;

    //置換表
    UctHashTable& hash_table_;

    const UsiOptions& usi_options_;

    //時間
    std::chrono::steady_clock::time_point start_;

    //ルート局面のインデックス
    Index root_index_;

    //時間制限(msec),ノード数制限
    int64_t time_limit_;
    int64_t node_limit_;

    GPUQueue& gpu_queue_;
    BackupQueue backup_queue_;
};

#endif //MIACIS_SEARCHER_HPP