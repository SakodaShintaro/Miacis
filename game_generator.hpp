#ifndef MIACIS_GAME_GENERATOR_HPP
#define MIACIS_GAME_GENERATOR_HPP

#include "replay_buffer.hpp"
#include "searcher_for_gen.hpp"
#include "game.hpp"

class GameGenerator{
public:
    GameGenerator(int64_t gpu_id, int64_t game_num, int64_t thread_num, ReplayBuffer& rb, NeuralNetwork<Tensor>& nn) :
    gpu_id_(gpu_id), thread_num_(thread_num), game_num_(game_num), rb_(rb) {
        clearEvalQueue();
    };

    //決まったゲーム数生成する関数
    void genGames();

    //SearchForGenに見せるqueue
    std::vector<float>& current_features_ = features_[0];
    std::vector<int32_t>& current_hash_index_queue_ = hash_index_queues_[0];
    std::vector<int32_t>& current_thread_ids_ = thread_ids_[0];

private:
    //生成してはreplay_bufferへ送る関数
    void genSlave();

    //GPUを回し続ける関数
    void gpuFunc();

    //キューをクリアする関数
    void clearEvalQueue();

    //使うGPUのid
    int64_t gpu_id_;

    //生成する局数
    std::atomic<int64_t> game_num_;

    //並列化するスレッド数
    int64_t thread_num_;

    //実行を続けるフラグ
    bool running_;

    //データを送るReplayBufferへの参照
    ReplayBuffer& rb_;

    //局面評価に用いるネットワーク
    NeuralNetwork<Tensor> evaluator_;

    //mutex
    std::vector<std::mutex> lock_node_;
    std::mutex lock_expand_;

    //キュー
    int8_t current_queue_index_;
    std::vector<float> features_[2];
    std::vector<int32_t> hash_index_queues_[2];
    std::vector<int32_t> thread_ids_[2];
};

#endif //MIACIS_GAME_GENERATOR_HPP