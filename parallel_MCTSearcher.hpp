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

class ParallelMCTSearcher {
public:
    //コンストラクタ
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
    
    //探索を行って一番良い指し手を返す関数
    Move think(Position& root);

private:
    static constexpr int32_t VIRTUAL_LOSS = 1;

    //再帰する探索関数
    ValueType uctSearch(Position& pos, Index current_index);

    //再帰しない探索関数
    void onePlay(Position& pos);

    //各スレッドに割り当てられる探索関数
    void parallelUctSearch(Position root);

    //ノードを展開する関数
    Index expandNode(Position& pos);

    //ノードを評価する関数
    void evalNode();

    //経過時間が持ち時間をオーバーしていないか確認する関数
    bool isTimeOver();

    //時間経過含め、playoutの回数なども考慮しplayoutを続けるかどうかを判定する関数
    bool shouldStop();

    //PVを取得する関数
    std::vector<Move> getPV() const;

    //情報をUSIプロトコルに従って標準出力に出す関数
    void printUSIInfo() const;

    //キューをクリア
    void clearEvalQueue();

    //置換表
    UctHashTable hash_table_;

    //Playout回数
    std::atomic<uint32_t> playout_num_;

    //ルート局面のインデックス
    Index current_root_index_;

    //時間
    std::chrono::steady_clock::time_point start_;

    //局面評価に用いるネットワーク
#ifdef USE_LIBTORCH
    NeuralNetwork evaluator_;
#else
    NeuralNetwork<Tensor>& evaluator_;
#endif

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

    //スレッド数
    int64_t thread_num_;

    //探索中であるかどうかのフラグ
    bool running_;
};

#endif // !PARALLEL_MCTSEARCHER_HPP