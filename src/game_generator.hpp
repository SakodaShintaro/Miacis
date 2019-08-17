#ifndef MIACIS_GAME_GENERATOR_HPP
#define MIACIS_GAME_GENERATOR_HPP

#include "replay_buffer.hpp"
#include "game.hpp"
#include "searcher_for_generate.hpp"
#include <atomic>
#include <mutex>
#include <stack>
#include <utility>

class GameGenerator {
public:
    GameGenerator(const UsiOptions& usi_options, FloatType Q_dist_lambda, ReplayBuffer& rb, NeuralNetwork evaluator)
        : usi_options_(usi_options), Q_dist_lambda_(Q_dist_lambda), rb_(rb), evaluator_(std::move(evaluator)) {
        evaluator_->eval();
    };

    //決まったゲーム数生成する関数
    void genGames(int64_t game_num);

    //mutex:AlphaZeroTrainerから触れるようにpublicに置いている
    std::mutex gpu_mutex;

private:
    //生成してはreplay_bufferへ送る関数
    void genSlave();

    //生成する局数
    std::atomic<int64_t> game_num_;

    //UsiOptionを持っておく
    const UsiOptions& usi_options_;

    //探索結果の分布として価値のsoftmax分布を混ぜる割合([0,1])
    //0で普通のAlphaZero
    const FloatType Q_dist_lambda_;

    //データを送るReplayBufferへの参照
    ReplayBuffer& rb_;

    //局面評価に用いるネットワーク
    NeuralNetwork evaluator_;
};

#endif //MIACIS_GAME_GENERATOR_HPP