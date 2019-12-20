#ifndef MIACIS_SEARCHER_FOR_GENERATE_HPP
#define MIACIS_SEARCHER_FOR_GENERATE_HPP

#include "searcher.hpp"
#include "game.hpp"
#include <stack>

class SearcherForGenerate : public Searcher {
public:
    //コンストラクタ
    SearcherForGenerate(int32_t id,
                        const UsiOptions& usi_options,
                        FloatType Q_dist_lambda,
                        std::vector<float>& input_queue,
                        std::vector<std::stack<int32_t>>& index_queue,
                        std::vector<std::stack<int32_t>>& action_queue,
                        std::vector<int32_t>& id_queue) :
            //ハッシュ容量は探索制限の2倍以上確保する
            //局面の遷移後も置換表を再利用しているため探索制限そのものだとまずい
            Searcher(usi_options.search_limit * 2, usi_options),
            id_(id),
            Q_dist_lambda_(Q_dist_lambda),
            input_queue_(input_queue), index_queue_(index_queue), action_queue_(action_queue), id_queue_(id_queue) {
        time_limit_ = LLONG_MAX;
        node_limit_ = usi_options.search_limit;
    }

private:
    //GameGeneratorでしか用いられないので全てprivateに置いてfriend指定をする
    friend class GameGenerator;

    //root局面を探索する準備を行う関数
    void prepareForCurrPos(Position& root);

    //探索1回を行う関数
    void select(Position& pos);

    //ノードを展開する関数
    Index expand(Position& pos, std::stack<int32_t>& indices, std::stack<int32_t>& actions);

    //現局面の探索結果を返す関数
    OneTurnElement resultForCurrPos(Position& root);

    //ディリクレ分布に従ったものを返す関数
    static std::vector<FloatType> dirichletDistribution(uint64_t k, FloatType alpha);

    //このスレッドのid
    int32_t id_;

    //探索結果の分布として価値のsoftmax分布を混ぜる割合([0,1])
    //0で普通のAlphaZero
    const FloatType Q_dist_lambda_;

    //評価要求を投げる先
    std::vector<float>& input_queue_;
    std::vector<std::stack<int32_t>>& index_queue_;
    std::vector<std::stack<int32_t>>& action_queue_;
    std::vector<int32_t>& id_queue_;

    //漸進的に更新されてしまうのでルート局面の生のValue出力を保存しておく
    //ルートノードのValueは更新する意味がないのでそのように変更すれば保存しておく必要もないのだが
    ValueType root_raw_value_;
};

#endif //MIACIS_SEARCHER_FOR_GENERATE_HPP