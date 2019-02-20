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
#ifdef USE_LIBTORCH
    GameGenerator(int64_t gpu_id, int64_t parallel_num, ReplayBuffer& rb, NeuralNetwork nn) :
            gpu_id_(gpu_id), parallel_num_(parallel_num), rb_(rb), evaluator_(std::move(nn)) {};
#else
    GameGenerator(int64_t gpu_id, int64_t parallel_num, ReplayBuffer& rb, std::shared_ptr<NeuralNetwork<Tensor>> nn) :
            gpu_id_(gpu_id), parallel_num_(parallel_num), rb_(rb), evaluator_(std::move(nn)) {};
#endif

    //決まったゲーム数生成する関数
    void genGames(int64_t game_num);

    //mutex:AlphaZeroTrainerから触れるようにpublicに置いている
    std::mutex gpu_mutex;

private:
    //使うGPUのid
    int64_t gpu_id_;

    //並列化する数
    int64_t parallel_num_;

    //データを送るReplayBufferへの参照
    ReplayBuffer& rb_;

    //局面評価に用いるネットワーク
#ifdef USE_LIBTORCH
    NeuralNetwork evaluator_;
#else
    std::shared_ptr<NeuralNetwork<Tensor>> evaluator_;
#endif

    //生成してはreplay_bufferへ送る関数
    void genSlave(int64_t id);

    //生成する局数
    std::atomic<int64_t> game_num_;

    //スレッド数:GPUを交互に使う"2"が最適値なはず
    static constexpr int32_t THREAD_NUM = 2;
};

#endif //MIACIS_GAME_GENERATOR_HPP