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
    GameGenerator(int64_t search_limit, int64_t draw_turn, int64_t thread_num, int64_t search_batch_size,
                  CalcType Q_dist_temperature, CalcType Q_dist_lambda, double C_PUCT, ReplayBuffer& rb, NeuralNetwork evaluator)
        : search_limit_(search_limit), draw_turn_(draw_turn),thread_num_(thread_num), search_batch_size_(search_batch_size),
          Q_dist_temperature_(Q_dist_temperature), Q_dist_lambda_(Q_dist_lambda), C_PUCT_(C_PUCT), rb_(rb), evaluator_(std::move(evaluator)) {
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

    //1局面あたりで行う探索回数
    int64_t search_limit_;

    //引き分け手数
    int64_t draw_turn_;

    //1GPUあたりに稼働するCPUのスレッド数
    int64_t thread_num_;

    //1CPUスレッドが1回GPUへ評価要求を投げるまでまとめて探索する回数
    int64_t search_batch_size_;

    //探索結果の分布として価値のsoftmax分布を計算するときの温度
    CalcType Q_dist_temperature_;

    //探索結果の分布として価値のsoftmax分布を混ぜる割合([0,1])
    //0で普通のAlphaZero
    CalcType Q_dist_lambda_;

    //探索クラスにおけるC_PUCT
    double C_PUCT_;

    //データを送るReplayBufferへの参照
    ReplayBuffer& rb_;

    //局面評価に用いるネットワーク
    NeuralNetwork evaluator_;
};

#endif //MIACIS_GAME_GENERATOR_HPP