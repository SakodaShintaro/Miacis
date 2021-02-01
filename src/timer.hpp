#ifndef MIACIS_TIMER_HPP
#define MIACIS_TIMER_HPP
#include <chrono>
#include <string>

class Timer {
public:
    Timer();
    void start();
    int64_t elapsedSeconds() const;
    std::string elapsedTimeStr() const;

private:
    std::chrono::steady_clock::time_point start_time_;
};

#endif //MIACIS_TIMER_HPP