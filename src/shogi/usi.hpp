#ifndef USI_HPP
#define USI_HPP

#include "../search_options.hpp"
#include "../searcher_for_play.hpp"
#include "position.hpp"
#include <functional>
#include <thread>

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
    void quit();
    void gameover();

private:
    std::unordered_map<std::string, std::function<void()>> command_;
    Position root_;
    std::unique_ptr<SearcherForPlay> searcher_;
    std::thread thread_;
    SearchOptions search_options_;
};

#endif