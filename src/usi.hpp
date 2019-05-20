#ifndef USI_HPP
#define USI_HPP

#include"position.hpp"
#include"searcher_for_play.hpp"
#include"usi_options.hpp"
#include<thread>
#include<string>
#include<functional>

class USI {
public:
    USI();
    void loop();
    void usi();
    void isready();
    void setoption();
    void usinewgame();
    void position();
    void go();
    void stop();
    void ponderhit();
    void quit();
    void gameover();
private:
    std::unordered_map<std::string, std::function<void()>> command_;
    Position root_;
    std::unique_ptr<SearcherForPlay> searcher_;
    std::thread thread_;
    USIOption usi_option_;
};

#endif