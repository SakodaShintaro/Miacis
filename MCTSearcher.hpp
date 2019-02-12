#ifndef MCTSEARCHER_HPP
#define MCTSEARCHER_HPP

#include"uct_hash_table.hpp"
#include"neural_network.hpp"
#include<vector>
#include<chrono>

//1スレッドでモンテカルロ木探索を行うクラス.マルチスレッド探索がちゃんと動いているか比較するために一応取っておくが,不要かもしれない
class MCTSearcher {
public:
    //コンストラクタ
#ifdef USE_LIBTORCH
    MCTSearcher(int64_t hash_size, NeuralNetwork nn) : hash_table_(hash_size), evaluator_(nn) {}
#else
    MCTSearcher(int64_t hash_size, NeuralNetwork<Tensor>& nn) : hash_table_(hash_size), evaluator_(nn) {}
#endif
    
    //探索を行って一番良い指し手を返す関数
    Move think(Position& root);

private:
    //再帰する探索関数
    ValueType uctSearch(Position& pos, Index current_index);

    //再帰しない探索関数
    void onePlay(Position& pos);

    //ノードを展開する関数
    Index expandNode(Position& pos);

    //ノードを評価する関数
    void evalNode(Position& pos, Index index);

    //経過時間が持ち時間をオーバーしていないか確認する関数
    bool isTimeOver();

    //時間経過含め、playoutの回数なども考慮しplayoutを続けるかどうかを判定する関数
    bool shouldStop();

    //PVを取得する関数
    std::vector<Move> getPV() const;

    //情報をUSIプロトコルに従って標準出力に出す関数
    void printUSIInfo() const;

    //置換表
    UctHashTable hash_table_;

    //ルート局面のインデックス
    Index current_root_index_;

    //時間
    std::chrono::steady_clock::time_point start_;

    //局面評価に用いるネットワーク
#ifdef USE_LIBTORCH
    NeuralNetwork evaluator_;
#else
    //学習には使わないのでTensor型のものだけを受け取れるようにする
    NeuralNetwork<Tensor>& evaluator_;
#endif
};

#endif // !MCTSEARCHER_HPP