#ifndef MIACIS_SEARCHER_HPP
#define MIACIS_SEARCHER_HPP

#include "hash_table.hpp"
#include "search_options.hpp"
#include <chrono>
#include <stack>

//GPUに対する評価要求を溜めるキュー
//hash_tableのindexに相当する入力inputを計算して適切な場所に書き込むことを要求する
struct GPUQueue {
    std::vector<float> inputs;
    std::vector<std::reference_wrapper<HashTable>> hash_tables;
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
    explicit Searcher(const SearchOptions& search_options, HashTable& hash_table, GPUQueue& gpu_queue)
        : hash_table_(hash_table), search_options_(search_options), gpu_queue_(gpu_queue) {}

    //再帰しない探索関数
    void select(Position& pos);

    //ノードを展開する関数
    Index expand(Position& pos, std::stack<int32_t>& indices, std::stack<int32_t>& actions);

    //バックアップ
    void backupAll();

    void clearBackupQueue() {
        backup_queue_.indices.clear();
        backup_queue_.actions.clear();
    }

private:
    //今のノードから遷移するべきノードを選択する関数
    int32_t selectMaxUcbChild(const HashEntry& node) const;

    //バックアップ
    void backup(std::stack<int32_t>& indices, std::stack<int32_t>& actions);

    //置換表
    HashTable& hash_table_;

    //探索のオプション
    const SearchOptions& search_options_;

    //評価要求を貯めるキュー。これは外部で生成して参照を貰う
    GPUQueue& gpu_queue_;

    //バックアップ要求を貯めるキュー。これは各インスタンスが生成して保持する
    BackupQueue backup_queue_;
};

#endif //MIACIS_SEARCHER_HPP