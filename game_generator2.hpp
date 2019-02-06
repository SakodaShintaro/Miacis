#ifndef MIACIS_GAME_GENERATOR2_HPP
#define MIACIS_GAME_GENERATOR2_HPP

#include "replay_buffer.hpp"
#include "game.hpp"
#include "uct_hash_table.hpp"
#include <atomic>
#include <mutex>
#include <stack>

class GameGenerator2{
public:
    GameGenerator2(int64_t gpu_id, int64_t searcher_num, ReplayBuffer &rb, NeuralNetwork<Tensor> &nn) :
    gpu_id_(gpu_id), searcher_num_(searcher_num), rb_(rb), evaluator_(nn) {};

    //決まったゲーム数生成する関数
    void genGames(int64_t game_num);

private:
    //生成してはreplay_bufferへ送る関数
    void genSlave(int64_t id);

    //1スレッドが持つsearcherの数
    int64_t searcher_num_;

    //使うGPUのid
    int64_t gpu_id_;

    //mutex
    std::mutex gpu_mutex_;

    //生成する局数
    std::atomic<int64_t> game_num_;

    //データを送るReplayBufferへの参照
    ReplayBuffer& rb_;

    //局面評価に用いるネットワーク
    NeuralNetwork<Tensor>& evaluator_;

    //探索クラス
    class SearcherForGen2 {
    public:
        //コンストラクタ
        SearcherForGen2(int64_t hash_size, int32_t id, std::vector<float> &features, std::vector<std::stack<int32_t>> &hash_indices,
                        std::vector<std::stack<int32_t>> &actions, std::vector<int32_t> &ids) : hash_table_(hash_size), id_(id),
                        features_(features), hash_indices_(hash_indices), actions_(actions), ids_(ids) {}

        //置換表:GameGeneratorが書き込める必要があるのでpublicに置く.friend指定とかでなんとかできるかも？
        UctHashTable hash_table_;

        //現局面の探索を終了して次局面へ遷移するかを判定する関数
        bool shouldGoNextPosition();

        //現局面の探索結果を返す関数
        std::pair<Move, TeacherType> resultForCurrPos(Position &root);

        //探索1回を行う関数
        void onePlay(Position &pos);

        //root局面を探索する準備を行う関数
        bool prepareForCurrPos(Position &root);

        void backup(std::stack<int32_t> &indices, std::stack<int32_t> &actions);
    private:
        //ノードを展開する関数
        Index expandNode(Position &pos, std::stack<int32_t> &indices, std::stack<int32_t> &actions);

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
        std::vector<float> &features_;
        std::vector<std::stack<int32_t>> &hash_indices_;
        std::vector<std::stack<int32_t>> &actions_;
        std::vector<int32_t> &ids_;
    };
};

#endif //MIACIS_GAME_GENERATOR_HPP