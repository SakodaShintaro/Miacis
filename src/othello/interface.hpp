#ifndef USI_HPP
#define USI_HPP

#include"position.hpp"
#include"../searcher_for_play.hpp"
#include"../usi_options.hpp"
#include<thread>
#include<functional>

class Interface {
public:
    Interface();
    void loop();
    void printOption();
    void setoption();

    //盤面を初期状態に戻す関数
    void init();

    //標準入力から行動を一つ受け取り、盤面を更新する関数
    void play();

    //現盤面について思考してbest_moveを標準出力に出し、盤面を更新する関数
    void go();
    void stop();
    void quit();
private:
    std::unordered_map<std::string, std::function<void()>> command_;
    Position root_;
    std::unique_ptr<SearcherForPlay> searcher_;
    std::thread thread_;
    UsiOptions options_;
};

#endif