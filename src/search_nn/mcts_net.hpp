#ifndef MIACIS_MCTS_NET_HPP
#define MIACIS_MCTS_NET_HPP

#include "hash_table_for_mcts_net.hpp"
#include "../search_options.hpp"

//MCTSを行うクラス
//想定の使い方は局面を放り投げて探索せよと投げることか
//なのでSearcherForPlayと置き換えられるように作れば良さそう
class MCTSNet {
public:
    explicit MCTSNet(const SearchOptions& search_options);

    //探索を行って一番良い指し手を返す関数
    Move think(Position& root, int64_t time_limit);

    const HashTableForMCTSNet& hashTable() { return hash_table_; }

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
    HashTableForMCTSNet hash_table_;

    //GPUは複数
    std::vector<NeuralNetwork> neural_networks_;
    std::vector<std::mutex> gpu_mutexes_;

    //時間
    std::chrono::steady_clock::time_point start_;

    //時間制限(msec),ノード数制限
    int64_t time_limit_;
    int64_t node_limit_;

    //次に表示する経過時間
    int64_t next_print_time_;

    //ログファイルを出力する場合のストリーム
    std::ofstream log_file_;

#ifdef SHOGI
    //定跡
    Book book_;
#endif
};

#endif //MIACIS_MCTS_NET_HPP