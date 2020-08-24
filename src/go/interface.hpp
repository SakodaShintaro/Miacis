#ifndef MIACIS_GO_INTERFACE_HPP
#define MIACIS_GO_INTERFACE_HPP

#include "../search_options.hpp"
#include "../searcher_for_play.hpp"
#include "position.hpp"
#include <functional>
#include <thread>

namespace Go {

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

    //盤面を初期状態に戻す関数
    void init();

    //Policyに従って対局し、Valueのログを残していく関数
    void outputValue();

    //現盤面について思考してbest_moveを標準出力に出し、盤面を更新する関数
    void go();
    void stop();
    void quit();

    //Go Text Protocolのコマンド
    void name();
    void version();
    void protocol_version();
    void list_commands();
    void komi();
    void boardsize();
    void clear_board();
    void genmove();
    void play();

private:
    std::unordered_map<std::string, std::function<void()>> command_;
    Position root_;
    std::unique_ptr<SearcherForPlay> searcher_;
    std::thread thread_;
    SearchOptions options_;
};

} // namespace Go

#endif //MIACIS_GO_INTERFACE_HPP