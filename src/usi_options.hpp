#ifndef MIACIS_USI_OPTIONS_HPP
#define MIACIS_USI_OPTIONS_HPP

#include "neural_network.hpp"
#include <cstdint>
#include <string>
#include <map>

struct SpinOption {
    SpinOption(int64_t& v, int64_t mi, int64_t ma) : value(v), min(mi), max(ma) {}
    int64_t& value;
    int64_t min, max;
};

struct StringOption {
    StringOption(std::string& v) : value(v) {}
    std::string& value;
};

class UsiOptions {
public:
    UsiOptions() {
        spin_options.emplace("USI_Hash",          SpinOption(USI_Hash          =       256, 0, INT32_MAX));
        spin_options.emplace("byoyomi_margin",    SpinOption(byoyomi_margin    =         0, 0, INT32_MAX));
        spin_options.emplace("random_turn",       SpinOption(random_turn       =         0, 0, INT32_MAX));
        spin_options.emplace("draw_turn",         SpinOption(draw_turn         =       512, 0, INT32_MAX));
        spin_options.emplace("print_interval",    SpinOption(print_interval    =       500, 1, INT32_MAX));
        spin_options.emplace("thread_num",        SpinOption(thread_num        =         2, 1, INT32_MAX));
        spin_options.emplace("search_batch_size", SpinOption(search_batch_size =       128, 1, INT32_MAX));
        spin_options.emplace("search_limit",      SpinOption(search_limit      = INT32_MAX, 1, INT32_MAX));
        spin_options.emplace("C_PUCT_x1000",      SpinOption(C_PUCT_x1000      =      1500, 1, INT32_MAX));
        spin_options.emplace("temperature_x1000", SpinOption(temperature_x1000 =         0, 0, INT32_MAX));
        spin_options.emplace("UCT_lambda_x1000",  SpinOption(UCT_lambda_x1000  =      1000, 0,      1000));
        spin_options.emplace("print_policy_num",  SpinOption(print_policy_num  =         0, 0,       593));
        string_options.emplace("model_name",      StringOption(model_name = NeuralNetworkImpl::DEFAULT_MODEL_NAME));
    }
    int64_t byoyomi_margin;
    int64_t random_turn;
    int64_t USI_Hash;
    int64_t draw_turn;
    int64_t print_interval;
    int64_t thread_num;
    int64_t search_batch_size;
    int64_t search_limit;
    int64_t C_PUCT_x1000;
    int64_t temperature_x1000;
    int64_t UCT_lambda_x1000;
    int64_t print_policy_num;
    std::string model_name;

    std::map<std::string, SpinOption> spin_options;
    std::map<std::string, StringOption> string_options;

private:
};


#endif //MIACIS_USI_OPTIONS_HPP
