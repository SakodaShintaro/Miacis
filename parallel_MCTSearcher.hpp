#ifndef PARALLEL_MCTSEARCHER_HPP
#define PARALLEL_MCTSEARCHER_HPP

#include"types.hpp"
#include"uct_hash_table.hpp"
#include"neural_network.hpp"
#include"usi_options.hpp"
#include"operate_params.hpp"
#include<vector>
#include<chrono>
#include<thread>
#include<atomic>
#include<stack>
#include<mutex>

//指定したスレッド数だけ実際にスレッドを作り並列化する(初期のdlshogiの実装)ならオンにし
//1スレッドが複数回計算要求を貯めていくねね将棋の実装にするならオフにする
//#define USE_PARALLEL_SEARCHER

class ParallelMCTSearcher {
public:
    //コンストラクタ
#ifdef USE_PARALLEL_SEARCHER

#ifdef USE_LIBTORCH
    ParallelMCTSearcher(int64_t hash_size, int64_t thread_num, NeuralNetwork nn) : hash_table_(hash_size),
    evaluator_(nn), thread_num_(thread_num) {
        lock_node_ = std::vector<std::mutex>(static_cast<unsigned long>(hash_table_.size()));
        clearEvalQueue();
    }
#else
    ParallelMCTSearcher(int64_t hash_size, int64_t thread_num, NeuralNetwork<Tensor>& nn) : hash_table_(hash_size),
    evaluator_(nn), thread_num_(thread_num) {
        lock_node_ = std::vector<std::mutex>(static_cast<unsigned long>(hash_table_.size()));
        clearEvalQueue();
    }
#endif

#else
#ifdef USE_LIBTORCH
    ParallelMCTSearcher(int64_t hash_size, int64_t thread_num, NeuralNetwork nn) : hash_table_(hash_size),
    evaluator_(nn), thread_num_(thread_num) {}
#else
    ParallelMCTSearcher(int64_t hash_size, int64_t thread_num, NeuralNetwork<Tensor>& nn) : hash_table_(hash_size),
    evaluator_(nn), thread_num_(thread_num) {}
#endif

#endif
    //探索を行って一番良い指し手を返す関数
    Move think(Position& root);

private:
    //--------------------------
    //    共通する変数,関数など
    //--------------------------
    //VIRTUAL_LOSSの大きさ
    static constexpr int32_t VIRTUAL_LOSS = 1;

    //経過時間が持ち時間をオーバーしていないか確認する関数
    bool isTimeOver();

    //時間経過含め、playoutの回数なども考慮しplayoutを続けるかどうかを判定する関数
    bool shouldStop();

    //スレッド数
    int64_t thread_num_;

    //置換表
    UctHashTable hash_table_;

    //ルート局面のインデックス
    Index current_root_index_;

    //Playout回数
    std::atomic<uint32_t> playout_num_;

    //PVを取得する関数
    std::vector<Move> getPV() const;

    //情報をUSIプロトコルに従って標準出力に出す関数
    void printUSIInfo() const;

    //時間
    std::chrono::steady_clock::time_point start_;

    //局面評価に用いるネットワーク
#ifdef USE_LIBTORCH
    NeuralNetwork evaluator_;
#else
    NeuralNetwork<Tensor>& evaluator_;
#endif

#ifdef USE_PARALLEL_SEARCHER

    //再帰する探索関数
    ValueType uctSearch(Position& pos, Index current_index);

    //再帰しない探索関数
    void onePlay(Position& pos);

    //ノードを展開する関数
    Index expandNode(Position& pos);

    //ノードを評価する関数
    void evalNode();

    //各スレッドに割り当てられる探索関数
    void parallelUctSearch(Position root);

    //キューをクリア
    void clearEvalQueue();

    //並列化に必要なもの
    //mutex
    std::vector<std::mutex> lock_node_;
    std::mutex lock_expand_;

    //キュー
    int8_t current_queue_index_;
    std::vector<float> features_[2];
    std::vector<Index> hash_index_queues_[2];
    std::vector<float>& current_features_ = features_[0];
    std::vector<Index>& current_hash_index_queue_ = hash_index_queues_[0];

    //探索中であるかどうかのフラグ
    bool running_;
#else
    static constexpr int32_t WORKER_NUM = 1;

    //各スレッドに割り当てられる探索関数
    void parallelUctSearch(Position root, int32_t id);

    //再帰しない探索関数
    void onePlay(Position& pos, int32_t id);

    //ノードを展開する関数
    Index expandNode(Position& pos, std::stack<int32_t>& indices, std::stack<int32_t>& actions, int32_t id);

    //バックアップ
    void backup(std::stack<int32_t> &indices, std::stack<int32_t> &actions);

    //キュー
    std::vector<float> input_queues_[WORKER_NUM];
    std::vector<Index> index_queues_[WORKER_NUM];
    std::vector<std::stack<Index>> route_queues_[WORKER_NUM];
    std::vector<std::stack<int32_t>> action_queues_[WORKER_NUM];
#endif
};

#endif // !PARALLEL_MCTSEARCHER_HPP