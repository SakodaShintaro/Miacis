#ifndef MIACIS_GAME_GENERATOR_HPP
#define MIACIS_GAME_GENERATOR_HPP

#include "replay_buffer.hpp"
#include "game.hpp"
#include "searcher_for_generate.hpp"
#include <atomic>
#include <mutex>
#include <stack>
#include <utility>

//自己対局をしてデータを生成するクラス
//一つのGPUに対して割り当てられる
class GameGenerator {
public:
    GameGenerator(const UsiOptions& usi_options, FloatType Q_dist_lambda, ReplayBuffer& rb, NeuralNetwork evaluator)
        : usi_options_(usi_options), Q_dist_lambda_(Q_dist_lambda), replay_buffer_(rb), neural_network_(std::move(evaluator)) {
        neural_network_->eval();
    };

    //決まったゲーム数生成する関数
    void genGames();

    //mutex:AlphaZeroTrainerから触れるようにpublicに置いている
    std::mutex gpu_mutex;

    bool stop;

private:
    //生成してはreplay_bufferへ送る関数
    void genSlave();

    //UsiOptionを持っておく
    const UsiOptions& usi_options_;

    //探索結果の分布として価値のsoftmax分布を混ぜる割合([0,1])
    //0で普通のAlphaZero
    const FloatType Q_dist_lambda_;

    //データを送るReplayBufferへの参照
    ReplayBuffer& replay_buffer_;

    //局面評価に用いるネットワーク
    NeuralNetwork neural_network_;

    GPUQueue gpu_queue_;
};

//一つのGPUに対して複数生成されるWorker
class GenerateWorker {
public:
    GenerateWorker(const UsiOptions& usi_options, FloatType Q_dist_lambda, ReplayBuffer& rb);
    void prepareForCurrPos();
    void select();
    void backup();
    OneTurnElement resultForCurrPos();

private:
    Game game_;
    Position position_;
    Searcher searcher_;
    UctHashTable hash_table_;
    Index root_index_;

    GPUQueue& gpu_queue_;

    //UsiOptionを持っておく
    const UsiOptions& usi_options_;

    //探索結果の分布として価値のsoftmax分布を混ぜる割合([0,1])
    //0で普通のAlphaZero
    const FloatType Q_dist_lambda_;

    //データを送るReplayBufferへの参照
    ReplayBuffer& replay_buffer_;
};

#endif //MIACIS_GAME_GENERATOR_HPP