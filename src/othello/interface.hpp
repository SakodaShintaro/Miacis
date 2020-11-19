#ifndef MIACIS_OTHELLO_INTERFACE_HPP
#define MIACIS_OTHELLO_INTERFACE_HPP

#include "../search_nn/models.hpp"
#include "../search_options.hpp"
#include "../searcher_for_play.hpp"
#include "position.hpp"
#include <functional>
#include <thread>

namespace Othello {

class Interface {
public:
    Interface();
    void loop();
    void printOption();
    void set();

    //テストで思考する関数
    void think();

    //テストで自己対局する関数
    void test();

    //テストとして自己対局を無限ループする関数
    void infiniteTest();

    //人間と対局する関数
    void battle();

    //ランダムプレイヤーと対局する関数
    void battleVSRandom();

    //パラメータを読み込んで対局
    void battleSelf();

    //盤面を初期状態に戻す関数
    void init();

    //標準入力から行動を一つ受け取り、盤面を更新する関数
    void play();

    //Policyに従って対局し、Valueのログを残していく関数
    void outputValue();

    //探索系のNNを用いて自己対局が正常に動くか検証する関数
    template<class T> void testSearchNN();

    //現盤面について思考してbest_moveを標準出力に出し、盤面を更新する関数
    void go();
    void stop();
    void quit();

private:
    std::unordered_map<std::string, std::function<void()>> command_;
    Position root_;
    std::unique_ptr<SearcherForPlay> searcher_;
    SimpleMLP simple_mlp_{ nullptr };
    SimpleLSTM simple_lstm_{ nullptr };
    MCTSNet mcts_net_{ nullptr };
    ProposedModelLSTM proposed_model_lstm_{ nullptr };
    ProposedModelTransformer transformer_model_{ nullptr };
    std::thread thread_;
    SearchOptions options_;
};

} // namespace Othello

#endif //MIACIS_OTHELLO_INTERFACE_HPP