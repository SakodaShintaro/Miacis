﻿#ifndef MIACIS_GAME_GENERATOR_HPP
#define MIACIS_GAME_GENERATOR_HPP

#include "replay_buffer.hpp"
#include "game.hpp"
#include "search_options.hpp"
#include "searcher.hpp"
#include <atomic>
#include <mutex>
#include <stack>
#include <utility>

//自己対局をしてデータを生成するクラス
//一つのGPUに対して割り当てられる
class GameGenerator {
public:
    GameGenerator(const SearchOptions& usi_options, int64_t worker_num, FloatType Q_dist_lambda, ReplayBuffer& rb, NeuralNetwork nn)
        : stop_signal(false), search_options_(usi_options), worker_num_(worker_num), Q_dist_lambda_(Q_dist_lambda), replay_buffer_(rb),
          neural_network_(std::move(nn)), gpu_queues_(search_options_.thread_num) {
        neural_network_->eval();
    };

    //生成してリプレイバッファに送り続ける関数
    void genGames();

    //mutex:AlphaZeroTrainerから触れるようにpublicに置いている
    std::mutex gpu_mutex;

    bool stop_signal;

private:
    //ディリクレ分布に従ったものを返す関数
    static std::vector<FloatType> dirichletDistribution(uint64_t k, FloatType alpha);

    //gpu_queue_に溜まっている入力を処理する関数
    void evalWithGPU(int64_t thread_id);

    //生成してはreplay_bufferへ送る関数
    void genSlave(int64_t thread_id);

    //UsiOptionを持っておく
    const SearchOptions& search_options_;

    //いくつのWorkerを並列に走らせるか
    const int64_t worker_num_;

    //探索結果の分布として価値のsoftmax分布を混ぜる割合([0,1])
    //0で普通のAlphaZero
    const FloatType Q_dist_lambda_;

    //データを送るReplayBufferへの参照
    ReplayBuffer& replay_buffer_;

    //局面評価に用いるネットワーク
    NeuralNetwork neural_network_;

    std::vector<GPUQueue> gpu_queues_;
};

//一つのGPUに対して複数生成されるWorker
class GenerateWorker {
public:
    GenerateWorker(const SearchOptions& usi_options, GPUQueue& gpu_queue, FloatType Q_dist_lambda, ReplayBuffer& rb);
    void prepareForCurrPos();
    void select();
    void backup();
    OneTurnElement resultForCurrPos();

private:
    //探索回数などを見て探索を続けるかどうかを判定する関数
    bool shouldStop();

    //UsiOptionを持っておく
    const SearchOptions& search_options_;

    //評価要求を投げる先
    GPUQueue& gpu_queue_;

    //探索結果の分布として価値のsoftmax分布を混ぜる割合([0,1])
    //0で普通のAlphaZero
    const FloatType Q_dist_lambda_;

    //データを送るReplayBufferへの参照
    ReplayBuffer& replay_buffer_;

    Game game_;
    Position position_;
    UctHashTable hash_table_;
    Searcher searcher_;

    //漸進的に更新されてしまうのでルート局面の生のValue出力を保存しておく
    //ルートノードのValueは更新する意味がないのでそのように変更すれば保存しておく必要もないのだが
    ValueType root_raw_value_;
};

#endif //MIACIS_GAME_GENERATOR_HPP