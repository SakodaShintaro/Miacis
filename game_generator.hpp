#ifndef MIACIS_GAME_GENERATOR_HPP
#define MIACIS_GAME_GENERATOR_HPP

#include "replay_buffer.hpp"
#include "game.hpp"
#include "uct_hash_table.hpp"
#include <atomic>
#include <mutex>

class GameGenerator{
public:
    GameGenerator(int64_t gpu_id, int64_t game_num, int64_t thread_num, ReplayBuffer& rb, NeuralNetwork<Tensor>& nn) :
    gpu_id_(gpu_id), thread_num_(thread_num), game_num_(game_num), rb_(rb), evaluator_(nn) {
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
    void genSlave(int64_t id);

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
    NeuralNetwork<Tensor>& evaluator_;

    //mutex
    std::mutex lock_expand_;

    //キュー
    int8_t current_queue_index_;
    std::vector<float> features_[2];
    std::vector<int32_t> hash_index_queues_[2];
    std::vector<int32_t> thread_ids_[2];

    //探索クラス
    class SearcherForGen {
    public:
        //コンストラクタ
        SearcherForGen(int64_t hash_size, int32_t id, GameGenerator& gg) : hash_table_(hash_size), id_(id), gg_(gg) {}

        //一番良い指し手と学習データを返す関数
        std::pair<Move, TeacherType> think(Position& root);

        //置換表:GameGeneratorが書き込める必要があるのでpublicに置く.friend指定とかでなんとかできるかも？
        UctHashTable hash_table_;

    private:
        //再帰する探索関数
        ValueType uctSearch(Position& pos, Index current_index);

        //プレイアウト1回
        void onePlay(Position& pos);

        //ノードを展開する関数
        Index expandNode(Position& pos);

        //時間経過含め、playoutの回数なども考慮しplayoutを続けるかどうかを判定する関数
        bool shouldStop();

        //Ucbを計算して最大値を持つインデックスを返す
        static int32_t selectMaxUcbChild(const UctHashEntry& current_node);

        //ディリクレ分布に従ったものを返す関数
        static std::vector<double> dirichletDistribution(int32_t k, double alpha);

        //評価要求を送る先
        GameGenerator& gg_;

        //このスレッドのid
        int32_t id_;

        //Playout回数
        uint32_t playout_num_;

        //ルート局面のインデックス
        Index current_root_index_;
    };
    std::vector<SearcherForGen> searchers_;
};

#endif //MIACIS_GAME_GENERATOR_HPP