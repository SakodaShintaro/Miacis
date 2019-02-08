#ifndef MCTSEARCHER_HPP
#define MCTSEARCHER_HPP

#include"types.hpp"
#include"uct_hash_table.hpp"
#include"neural_network.hpp"
#include"usi_options.hpp"
#include"operate_params.hpp"
#include<vector>
#include<chrono>
#include<atomic>
#include<stack>

//1スレッドでモンテカルロ木探索を行うクラス.マルチスレッド探索がちゃんと動いているか比較するために一応取っておくが,不要かもしれない
class MCTSearcher {
public:
    //コンストラクタ
#ifdef USE_LIBTORCH
    MCTSearcher(int64_t hash_size, NeuralNetwork nn) : hash_table_(hash_size), evaluator_(nn) {}
#else
    MCTSearcher(int64_t hash_size, NeuralNetwork<Tensor>& nn) : hash_table_(hash_size), evaluator_(nn) {}
#endif
    
    //一番良い指し手と学習データを返す関数
    std::pair<Move, TeacherType> think(Position& root);

private:
    //再帰する探索関数
    ValueType uctSearch(Position& pos, Index current_index);

    //プレイアウト1回
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

    //Ucbを計算して最大値を持つインデックスを返す
    static int32_t selectMaxUcbChild(const UctHashEntry& current_node);

    //ディリクレ分布に従ったものを返す関数
    static std::vector<double> dirichletDistribution(int32_t k, double alpha);

    //置換表
    UctHashTable hash_table_;

    //Playout回数
    uint32_t playout_num;

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