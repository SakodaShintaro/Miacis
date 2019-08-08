#ifndef USI_HPP
#define USI_HPP

#include"position.hpp"
#include"searcher_for_play.hpp"
#include<thread>
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

    struct USIOption {
        int64_t byoyomi_margin;
        int64_t random_turn;
        int64_t USI_Hash;
        int64_t draw_turn;
        int64_t print_interval;
        int64_t thread_num;
        int64_t search_batch_size;
        int64_t search_limit;
        int64_t C_PUCT_1000;
        int64_t temperature_x1000;
        int64_t UCT_lambda_x1000;
        int64_t print_policy_num;
        std::string model_name;
    } usi_option_;
};

#endif