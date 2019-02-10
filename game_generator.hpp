#include <utility>

#ifndef MIACIS_GAME_GENERATOR_HPP
#define MIACIS_GAME_GENERATOR_HPP

#include "replay_buffer.hpp"
#include "game.hpp"
#include "uct_hash_table.hpp"
#include <atomic>
#include <mutex>
#include <stack>

//#define USE_PARALLEL_SEARCHER

class GameGenerator {
public:
#ifdef USE_LIBTORCH
    GameGenerator(int64_t gpu_id, int64_t parallel_num, ReplayBuffer& rb, NeuralNetwork nn) :
            gpu_id_(gpu_id), parallel_num_(parallel_num), rb_(rb), evaluator_(std::move(nn)) {};
#else
    GameGenerator(int64_t gpu_id, int64_t parallel_num, ReplayBuffer& rb, NeuralNetwork<Tensor>& nn) :
            gpu_id_(gpu_id), parallel_num_(parallel_num), rb_(rb), evaluator_(nn) {};
#endif

    //決まったゲーム数生成する関数
    void genGames(int64_t game_num);

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
    NeuralNetwork<Tensor>& evaluator_;
#endif

#ifdef USE_PARALLEL_SEARCHER
    //生成してはreplay_bufferへ送る関数
    void genSlave(int64_t id);

    //GPUを回し続ける関数
    void gpuFunc();

    //キューをクリアする関数
    void clearEvalQueue();

    //生成する局数
    std::atomic<int64_t> game_num_;

    //実行を続けるフラグ
    bool running_;

    //mutex
    std::mutex lock_expand_;

    //キュー
    int8_t current_queue_index_;
    std::vector<float> features_[2];
    std::vector<int32_t> hash_index_queues_[2];
    std::vector<int32_t> thread_ids_[2];
    //SearchForGenに見せるqueue
    std::vector<float>& current_features_ = features_[0];
    std::vector<int32_t>& current_hash_index_queue_ = hash_index_queues_[0];
    std::vector<int32_t>& current_thread_ids_ = thread_ids_[0];

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
#else

    //生成してはreplay_bufferへ送る関数
    void genSlave(int64_t id);

    //mutex
    std::mutex gpu_mutex_;

    //生成する局数
    std::atomic<int64_t> game_num_;

    //スレッド数:GPUを交互に使う"2"が最適値なはず
    static constexpr int32_t THREAD_NUM = 2;

    //探索クラス
    class SearcherForGen {
    public:
        //コンストラクタ
        SearcherForGen(int64_t hash_size, int32_t id, std::vector<float>& features,
                       std::vector<std::stack<int32_t>>& hash_indices,
                       std::vector<std::stack<int32_t>>& actions, std::vector<int32_t>& ids) :
                hash_table_(hash_size), id_(id), features_(features), hash_indices_(hash_indices), actions_(actions),
                ids_(ids) {}

        //置換表:GameGeneratorが書き込める必要があるのでpublicに置く.friend指定とかでなんとかできるかも？
        UctHashTable hash_table_;

        //現局面の探索を終了して次局面へ遷移するかを判定する関数
        bool shouldStop();

        //現局面の探索結果を返す関数
        std::pair<Move, TeacherType> resultForCurrPos(Position& root);

        //探索1回を行う関数
        void onePlay(Position& pos);

        //root局面を探索する準備を行う関数
        bool prepareForCurrPos(Position& root);

        //GPUの計算結果をルートノードまでバックアップする関数
        void backup(std::stack<int32_t>& indices, std::stack<int32_t>& actions);

    private:
        //ノードを展開する関数
        Index expandNode(Position& pos, std::stack<int32_t>& indices, std::stack<int32_t>& actions);

        //Ucbを計算して最大値を持つインデックスを返す
        static int32_t selectMaxUcbChild(const UctHashEntry& current_node);

        //ディリクレ分布に従ったものを返す関数
        static std::vector<double> dirichletDistribution(int32_t k, double alpha);

        //このスレッドのid
        int32_t id_;

        //Playout回数
        uint32_t playout_num_;

        //ルート局面のインデックス
        Index current_root_index_;

        //評価要求を投げる先
        std::vector<float>& features_;
        std::vector<std::stack<int32_t>>& hash_indices_;
        std::vector<std::stack<int32_t>>& actions_;
        std::vector<int32_t>& ids_;
    };

#endif
};

#endif //MIACIS_GAME_GENERATOR_HPP