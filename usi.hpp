#ifndef USI_HPP
#define USI_HPP

#include"common.hpp"
#include"position.hpp"
#include"searcher.hpp"
#include"move.hpp"
#include<thread>
#include<string>

class USI {
public:
    USI() : root_(*eval_params) {}
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
    Position root_;
    std::thread thread_;
};

#endif
