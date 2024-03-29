#ifndef MIACIS_SEARCH_OPTIONS_HPP
#define MIACIS_SEARCH_OPTIONS_HPP

#include "../model/model_common.hpp"
#include <cstdint>
#include <cuda_runtime_api.h>
#include <map>
#include <string>

struct CheckOption {
    explicit CheckOption(bool& v) : value(v) {}
    bool& value;
};

struct SpinOption {
    SpinOption(int64_t& v, int64_t mi, int64_t ma) : value(v), min(mi), max(ma) {}
    int64_t& value;
    int64_t min, max;
};

struct FilenameOption {
    explicit FilenameOption(std::string& v) : value(v) {}
    std::string& value;
};

struct SearchOptions {
public:
    SearchOptions() {
        //上限ギリギリだとオーバーフローしかねないので適当な値で収める
        constexpr int32_t MAX = 1e9;
        //GPUの数を取得。GPUがない場合もとりあえずここでは1として、内部的にCPUで計算する
        int gpu;
        cudaGetDeviceCount(&gpu);
        // clang-format off
        check_options.emplace("USI_Ponder",              CheckOption(USI_Ponder = false));
        check_options.emplace("leave_root",              CheckOption(leave_root = true));
        check_options.emplace("use_book",                CheckOption(use_book   = false));
        check_options.emplace("print_info",              CheckOption(print_info = true));
        check_options.emplace("output_log_file",         CheckOption(output_log_file = false));
        spin_options.emplace("USI_Hash",                 SpinOption(USI_Hash                 =   256,    0,  MAX));
        spin_options.emplace("byoyomi_margin",           SpinOption(byoyomi_margin           =     0,    0,  MAX));
        spin_options.emplace("random_turn",              SpinOption(random_turn              =     0,    0,  MAX));
        spin_options.emplace("draw_turn",                SpinOption(draw_turn                =   320,    0,  MAX));
        spin_options.emplace("print_interval",           SpinOption(print_interval           =   500,    1,  MAX));
        spin_options.emplace("gpu_num",                  SpinOption(gpu_num                  =   gpu,    1,  gpu));
        spin_options.emplace("thread_num_per_gpu",       SpinOption(thread_num_per_gpu       =     2,    1,  MAX));
        spin_options.emplace("search_batch_size",        SpinOption(search_batch_size        =    64,    1,  MAX));
        spin_options.emplace("search_limit",             SpinOption(search_limit             =   MAX,    1,  MAX));
#ifdef USE_CATEGORICAL
        spin_options.emplace("Q_coeff_x1000",            SpinOption(Q_coeff_x1000            =     0,    0,  MAX));
        spin_options.emplace("FPU_x1000",                SpinOption(FPU_x1000                =     0, -MAX,  MAX));
#else
        spin_options.emplace("Q_coeff_x1000",            SpinOption(Q_coeff_x1000            =  1000,    0,  MAX));
        spin_options.emplace("FPU_x1000",                SpinOption(FPU_x1000                = -1000, -MAX,  MAX));
#endif
        spin_options.emplace("C_PUCT_x1000",             SpinOption(C_PUCT_x1000             =  2500,    1,  MAX));
        spin_options.emplace("P_coeff_x1000",            SpinOption(P_coeff_x1000            =  1000,    0,  MAX));
        spin_options.emplace("virtual_loss",             SpinOption(virtual_loss             =     1,    1,  MAX));
        spin_options.emplace("temperature_x1000",        SpinOption(temperature_x1000        =     0,    0,  MAX));
        spin_options.emplace("policy_temperature_x1000", SpinOption(policy_temperature_x1000 =  1000,    1,  MAX));
        spin_options.emplace("book_temperature_x1000",   SpinOption(book_temperature_x1000   =     0,    0, 1000));
        spin_options.emplace("UCT_lambda_x1000",         SpinOption(UCT_lambda_x1000         =  1000,    0, 1000));
        spin_options.emplace("print_policy_num",         SpinOption(print_policy_num         =     0,    0,  593));
        spin_options.emplace("remained_turn_divisor",    SpinOption(remained_turn_divisor    =     1,    1,  MAX));
        spin_options.emplace("hold_moves_num",           SpinOption(hold_moves_num           =    32,    1,  593));
        filename_options.emplace("model_name",           FilenameOption(model_name = DEFAULT_ENGINE_NAME));
        filename_options.emplace("book_file_name",       FilenameOption(book_file_name       = "book.txt"));
        // clang-format on
    }
    bool USI_Ponder;
    bool leave_root;
    bool use_book;
    bool print_info;
    bool output_log_file;
    int64_t byoyomi_margin;
    int64_t random_turn;
    int64_t USI_Hash;
    int64_t draw_turn;
    int64_t print_interval;
    int64_t gpu_num;
    int64_t thread_num_per_gpu;
    int64_t search_batch_size;
    int64_t search_limit;
    int64_t Q_coeff_x1000;
    int64_t C_PUCT_x1000;
    int64_t P_coeff_x1000;
    int64_t FPU_x1000;
    int64_t virtual_loss;
    int64_t temperature_x1000;
    int64_t policy_temperature_x1000;
    int64_t book_temperature_x1000;
    int64_t UCT_lambda_x1000;
    int64_t print_policy_num;
    int64_t remained_turn_divisor;
    int64_t hold_moves_num;
    std::string model_name;
    std::string book_file_name;

    std::map<std::string, CheckOption> check_options;
    std::map<std::string, SpinOption> spin_options;
    std::map<std::string, FilenameOption> filename_options;
};

#endif //MIACIS_SEARCH_OPTIONS_HPP