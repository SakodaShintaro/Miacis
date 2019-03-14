#include <utility>

#ifndef MIACIS_SEARCHER_FOR_PLAY_HPP
#define MIACIS_SEARCHER_FOR_PLAY_HPP

#include"searcher.hpp"
#include"neural_network.hpp"
#include<stack>
#include<mutex>

class SearcherForPlay : public Searcher {
public:
#ifdef USE_LIBTORCH
    SearcherForPlay(int64_t hash_size, uint64_t thread_num, uint64_t search_batch_size, NeuralNetwork nn) :
    Searcher(hash_size), thread_num_(thread_num), search_batch_size_(search_batch_size), evaluator_(std::move(nn)) {
        lock_node_ = std::vector<std::mutex>(hash_table_.size());
        input_queues_.resize(thread_num);
        index_queues_.resize(thread_num);
        route_queues_.resize(thread_num);
        action_queues_.resize(thread_num);
        redundancy_num_.resize(thread_num);
    }
#else
    SearcherForPlay(int64_t hash_size, uint64_t thread_num, uint64_t search_batch_size, std::shared_ptr<NeuralNetwork<Tensor>> nn) :
    Searcher(hash_size), thread_num_(thread_num), search_batch_size_(search_batch_size), evaluator_(std::move(nn)) {
        lock_node_ = std::vector<std::mutex>(hash_table_.size());
        input_queues_.resize(thread_num);
        index_queues_.resize(thread_num);
        route_queues_.resize(thread_num);
        action_queues_.resize(thread_num);
        redundancy_num_.resize(thread_num);
    }
#endif

    //探索を行って一番良い指し手を返す関数
    Move think(Position& root);

private:
    //--------------------------
    //    共通する変数,関数など
    //--------------------------
    //VIRTUAL_LOSSの大きさ
    static constexpr int32_t VIRTUAL_LOSS = 1;

    //PVを取得する関数
    std::vector<Move> getPV() const;

    //情報をUSIプロトコルに従って標準出力に出す関数
    void printUSIInfo() const;

    //各スレッドに割り当てられる探索関数
    void parallelUctSearch(Position root, int32_t id);

    //再帰しない探索関数
    void select(Position& pos, int32_t id);

    //ノードを展開する関数
    Index expand(Position& pos, std::stack<int32_t>& indices, std::stack<int32_t>& actions, int32_t id);

    //バックアップ
    void backup(std::stack<int32_t>& indices, std::stack<int32_t>& actions, int32_t add_num);

    //局面評価に用いるネットワーク
#ifdef USE_LIBTORCH
    NeuralNetwork evaluator_;
#else
    std::shared_ptr<NeuralNetwork<Tensor>> evaluator_;
#endif

    //mutex
    std::vector<std::mutex> lock_node_;
    std::mutex lock_expand_;

    //スレッド数
    uint64_t thread_num_;

    //各スレッドが1回GPUを使うまでに探索する数
    uint64_t search_batch_size_;

    //キュー
    std::vector<std::vector<float>> input_queues_;
    std::vector<std::vector<Index>> index_queues_;
    std::vector<std::vector<std::stack<Index>>> route_queues_;
    std::vector<std::vector<std::stack<int32_t>>> action_queues_;
    std::vector<std::vector<int32_t>> redundancy_num_;
};

#endif //MIACIS_SEARCHER_FOR_PLAY_HPP