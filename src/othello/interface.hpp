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
    void play();
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