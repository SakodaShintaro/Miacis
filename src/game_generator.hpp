#ifndef MIACIS_GAME_GENERATOR_HPP
#define MIACIS_GAME_GENERATOR_HPP

#include "replay_buffer.hpp"
#include "game.hpp"
#include "uct_hash_table.hpp"
#include "searcher_for_generate.hpp"
#include <atomic>
#include <mutex>
#include <stack>
#include <utility>

class GameGenerator {
public:
    GameGenerator(int64_t search_limit, int64_t draw_turn, int64_t thread_num, int64_t search_batch_size,
                  ReplayBuffer& rb, NeuralNetwork evaluator)
        : search_limit_(search_limit), draw_turn_(draw_turn),thread_num_(thread_num),
          search_batch_size_(search_batch_size), rb_(rb), evaluator_(std::move(evaluator)) {
        evaluator_->eval();
    };

    //決まったゲーム数生成する関数
    void genGames(int64_t game_num);

    //mutex:AlphaZeroTrainerから触れるようにpublicに置いている
    std::mutex gpu_mutex;

private:
    //生成してはreplay_bufferへ送る関数
    void genSlave(int64_t id);

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

    //データを送るReplayBufferへの参照
    ReplayBuffer& rb_;

    //局面評価に用いるネットワーク
    NeuralNetwork evaluator_;
};

#endif //MIACIS_GAME_GENERATOR_HPP