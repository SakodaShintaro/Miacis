#ifndef MIACIS_SEARCHER_FOR_PLAY_HPP
#define MIACIS_SEARCHER_FOR_PLAY_HPP

#include"searcher.hpp"
#include"neural_network.hpp"
#include<stack>
#include<mutex>

class SearcherForPlay {
public:
    SearcherForPlay(const UsiOptions& usi_options);

    //探索を行って一番良い指し手を返す関数
    Move think(Position& root, int64_t time_limit);

private:
    //--------------------------
    //    共通する変数,関数など
    //--------------------------
    //VIRTUAL_LOSSの大きさ
    static constexpr int32_t VIRTUAL_LOSS = 1;

    //時間制限含め探索を続けるかどうかを判定する関数
    bool shouldStop();

    //GPUに付随するスレッド
    void gpuThreadFunc(const Position& root, int64_t gpu_id);

    //各GPUの下で動くスレッド
    void workerThreadFunc(Position root, int64_t gpu_id, int64_t thread_id);

    //PVを取得する関数
    std::vector<Move> getPV() const;

    //情報をUSIプロトコルに従って標準出力に出す関数
    void printUSIInfo() const;

    const UsiOptions& usi_options_;

    //置換表は1個
    UctHashTable hash_table_;

    //GPUは複数
    std::vector<NeuralNetwork> neural_networks_;
    std::vector<std::mutex> gpu_mutexes_;

    //1つのGPUに対してgpu_queue,searcherを複数
    std::vector<std::vector<GPUQueue>> gpu_queues_;
    std::vector<std::vector<Searcher>> searchers_;

    //ルート局面のインデックス
    Index root_index_;

    //時間
    std::chrono::steady_clock::time_point start_;

    //時間制限(msec),ノード数制限
    int64_t time_limit_;
    int64_t node_limit_;

    //次に表示する経過時間
    int64_t next_print_time_;
};

#endif //MIACIS_SEARCHER_FOR_PLAY_HPP