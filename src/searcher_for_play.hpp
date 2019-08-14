#ifndef MIACIS_SEARCHER_FOR_PLAY_HPP
#define MIACIS_SEARCHER_FOR_PLAY_HPP

#include"searcher.hpp"
#include"neural_network.hpp"
#include"usi_options.hpp"
#include<stack>
#include<mutex>

class SearcherForPlay : public Searcher {
public:
    SearcherForPlay(const UsiOptions& usi_options, NeuralNetwork evaluator) :
            //ハッシュ容量はMByte単位になっているので個数に変換する
            Searcher(usi_options.USI_Hash * 1024 * 1024 / 10000, usi_options.C_PUCT_x1000 / 1000.0),
            usi_options_(usi_options),
            next_print_time_(LLONG_MAX),
            evaluator_(std::move(evaluator)) {
        lock_node_ = std::vector<std::mutex>(hash_table_.size());
        input_queues_.resize(usi_options.thread_num);
        index_queues_.resize(usi_options.thread_num);
        route_queues_.resize(usi_options.thread_num);
        action_queues_.resize(usi_options.thread_num);
    }

    //探索を行って一番良い指し手を返す関数
    Move think(Position& root, int64_t time_limit);

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

    //mutex
    std::vector<std::mutex> lock_node_;
    std::mutex lock_all_table_;
    std::mutex lock_gpu_;

    //複数のオプションをバラバラにまた設定するのではなくUsiオプションへのconst参照をまるごと持ってしまうのがわかりやすそう
    //多少探索には関係ないものもあるけど、どうせUsiオプションで設定するものってほとんどが探索で用いるパラメータ等の設定だし
    const UsiOptions& usi_options_;

    //次に表示する経過時間
    int64_t next_print_time_;

    //局面評価に用いるネットワーク
    NeuralNetwork evaluator_;

    //キュー
    std::vector<std::vector<float>> input_queues_;
    std::vector<std::vector<Index>> index_queues_;
    std::vector<std::vector<std::stack<Index>>> route_queues_;
    std::vector<std::vector<std::stack<int32_t>>> action_queues_;
};

#endif //MIACIS_SEARCHER_FOR_PLAY_HPP