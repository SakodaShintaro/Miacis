#include"base_trainer.hpp"

void BaseTrainer::timestamp() {
    auto elapsed = std::chrono::steady_clock::now() - start_time_;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    auto minutes = seconds / 60;
    seconds %= 60;
    auto hours = minutes / 60;
    minutes %= 60;
    std::cout << std::setfill('0') << std::setw(3) << hours << ":"
        << std::setfill('0') << std::setw(2) << minutes << ":"
        << std::setfill('0') << std::setw(2) << seconds;
    log_file_ << std::setfill('0') << std::setw(3) << hours << ":"
        << std::setfill('0') << std::setw(2) << minutes << ":"
        << std::setfill('0') << std::setw(2) << seconds;
}

bool BaseTrainer::isLegalOptimizer() {
    return (OPTIMIZER_NAME == "SGD"
        || OPTIMIZER_NAME == "MOMENTUM");
}