#ifndef MIACIS_SEARCHER_FOR_GENERATE_HPP
#define MIACIS_SEARCHER_FOR_GENERATE_HPP

#include "searcher.hpp"
#include <stack>

class SearcherForGenerate : public Searcher {
private:
    //GameGeneratorでしか用いられないので全てprivateに置いてfriend指定をする
    friend class GameGenerator;

    //コンストラクタ
    SearcherForGenerate(int64_t hash_size, int32_t id, std::vector<float>& input_queue, std::vector<std::stack<int32_t>>& index_queue,
            std::vector<std::stack<int32_t>>& action_queue, std::vector<int32_t>& id_queue) :
    Searcher(hash_size), id_(id), input_queue_(input_queue), index_queue_(index_queue), action_queue_(action_queue),
    id_queue_(id_queue) {}

    //root局面を探索する準備を行う関数
    bool prepareForCurrPos(Position& root);

    //探索1回を行う関数
    void onePlay(Position& pos);

    //ノードを展開する関数
    Index expandNode(Position& pos, std::stack<int32_t>& indices, std::stack<int32_t>& actions);

    //GPUの計算結果をルートノードまでバックアップする関数
    void backup(std::stack<int32_t>& indices, std::stack<int32_t>& actions);

    //現局面の探索結果を返す関数
    std::pair<Move, TeacherType> resultForCurrPos(Position& root);

    //ディリクレ分布に従ったものを返す関数
    static std::vector<double> dirichletDistribution(uint64_t k, double alpha);

    //詰み探索
    bool mateSearch(Position pos, int32_t depth);

    //このスレッドのid
    int32_t id_;

    //評価要求を投げる先
    std::vector<float>& input_queue_;
    std::vector<std::stack<int32_t>>& index_queue_;
    std::vector<std::stack<int32_t>>& action_queue_;
    std::vector<int32_t>& id_queue_;
};

#endif //MIACIS_SEARCHER_FOR_GENERATE_HPP