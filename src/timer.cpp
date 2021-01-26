#include "timer.hpp"
#include <iomanip>
#include <sstream>

Timer::Timer() { start(); }

void Timer::start() { start_time_ = std::chrono::steady_clock::now(); }

int64_t Timer::elapsedSeconds() const {
    auto elapsed = std::chrono::steady_clock::now() - start_time_;
    int64_t seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    return seconds;
}

std::string Timer::elapsedTimeStr() const {
    int64_t seconds = elapsedSeconds();

    std::stringstream ss;

    //hhhh:mm:ssで文字列化
    int64_t minutes = seconds / 60;
    seconds %= 60;
    int64_t hours = minutes / 60;
    minutes %= 60;
    ss << std::setfill('0') << std::setw(4) << hours << ":" << std::setfill('0') << std::setw(2) << minutes << ":"
       << std::setfill('0') << std::setw(2) << seconds;
    return ss.str();
}