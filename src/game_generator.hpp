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
    GameGenerator(ReplayBuffer& rb, NeuralNetwork nn) :
            rb_(rb), evaluator_(std::move(nn)) {
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

    //データを送るReplayBufferへの参照
    ReplayBuffer& rb_;

    //局面評価に用いるネットワーク
    NeuralNetwork evaluator_;
};

#endif //MIACIS_GAME_GENERATOR_HPP