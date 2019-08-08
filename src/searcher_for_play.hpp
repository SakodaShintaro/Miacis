#ifndef MIACIS_SEARCHER_FOR_PLAY_HPP
#define MIACIS_SEARCHER_FOR_PLAY_HPP

#include"searcher.hpp"
#include"neural_network.hpp"
#include<stack>
#include<mutex>

class SearcherForPlay : public Searcher {
public:
    SearcherForPlay(int64_t hash_size, FloatType C_PUCT, uint64_t thread_num, uint64_t search_batch_size,
                    NeuralNetwork evaluator, FloatType temperature, FloatType lambda, int64_t print_policy_num,
                    int64_t draw_turn) :
            Searcher(hash_size, C_PUCT), evaluator_(std::move(evaluator)), thread_num_(thread_num), search_batch_size_(search_batch_size),
            print_interval_(LLONG_MAX), next_print_time_(LLONG_MAX), temperature_(temperature), lambda_(lambda),
            print_policy_num_(print_policy_num), draw_turn_(draw_turn) {
        lock_node_ = std::vector<std::mutex>(hash_table_.size());
        input_queues_.resize(thread_num);
        index_queues_.resize(thread_num);
        route_queues_.resize(thread_num);
        action_queues_.resize(thread_num);
    }

    //探索を行って一番良い指し手を返す関数
    Move think(Position& root, int64_t time_limit, int64_t node_limit, int64_t random_turn, int64_t print_interval);

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
    void backup(std::stack<int32_t>& indices, std::stack<int32_t>& actions);

    //局面評価に用いるネットワーク
    NeuralNetwork evaluator_;

    //mutex
    std::vector<std::mutex> lock_node_;
    std::mutex lock_all_table_;
    std::mutex lock_gpu_;

    //スレッド数
    uint64_t thread_num_;

    //各スレッドが1回GPUを使うまでに探索する数
    uint64_t search_batch_size_;

    //表示間隔.厳密には取れないので適当な間隔で表示する
    int64_t print_interval_;
    int64_t next_print_time_;

    //ソフトマックス分布の温度: 0のときは行動分布に従った選択をする
    const FloatType temperature_;

    //Sarsa-UCT(λ)のλ
    const FloatType lambda_;

    //上位n手までの方策を出力
    const int64_t print_policy_num_;

    //引き分けとする手数
    const int64_t draw_turn_;

    //キュー
    std::vector<std::vector<float>> input_queues_;
    std::vector<std::vector<Index>> index_queues_;
    std::vector<std::vector<std::stack<Index>>> route_queues_;
    std::vector<std::vector<std::stack<int32_t>>> action_queues_;
};

#endif //MIACIS_SEARCHER_FOR_PLAY_HPP