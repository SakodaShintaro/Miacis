#ifndef USI_HPP
#define USI_HPP

#include"position.hpp"
#include<thread>
#include<string>

class USI {
public:
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