#ifndef MIACIS_SEARCHER_FOR_PLAY_HPP
#define MIACIS_SEARCHER_FOR_PLAY_HPP

#include "model/infer_model.hpp"
#include "model/model_common.hpp"
#include "searcher.hpp"
#include "searcher_for_mate.hpp"
#include <mutex>
#include <stack>

#ifdef SHOGI
#include "shogi/book.hpp"
#endif

class SearcherForPlay {
public:
    explicit SearcherForPlay(const SearchOptions& search_options);

    //探索を行って一番良い指し手を返す関数
    Move think(Position& root, int64_t time_limit);

    const HashTable& hashTable() { return hash_table_; }

    //探索の終了判定。外部から探索を止めたい場合にはこれをtrueにする
    bool stop_signal;

private:
    //時間制限含め探索を続けるかどうかを判定する関数
    bool shouldStop();

    //GPUに付随するスレッド。内部的に下のworkerThreadFuncをthread_num_per_gpu数だけ生成する
    void gpuThreadFunc(const Position& root, int64_t gpu_id);

    //各GPUの下で動くスレッド
    void workerThreadFunc(Position root, int64_t gpu_id, int64_t thread_id);

    //PVを取得する関数
    std::vector<Move> getPV() const;

    //探索結果をストリームに出力する関数
    void outputInfo(std::ostream& ost, int64_t gather_num) const;

    //探索に関するオプション
    const SearchOptions& search_options_;

    //置換表は1個
    HashTable hash_table_;

    //GPUは複数
    std::vector<InferModel> neural_networks_;
    std::vector<std::mutex> gpu_mutexes_;

    //1つのGPUに対してgpu_queue,searcherを複数
    std::vector<std::vector<GPUQueue>> gpu_queues_;
    std::vector<std::vector<Searcher>> searchers_;

    //時間
    std::chrono::steady_clock::time_point start_;

    //時間制限(msec),ノード数制限
    int64_t time_limit_{};
    int64_t node_limit_{};

    //次に表示する経過時間
    int64_t next_print_time_{};

    //詰み探索エージェント
    SearcherForMate mate_searcher_;

    //ログファイルを出力する場合のストリーム
    std::ofstream log_file_;

#ifdef SHOGI
    //定跡
    Book book_;
#endif
};

#endif //MIACIS_SEARCHER_FOR_PLAY_HPP