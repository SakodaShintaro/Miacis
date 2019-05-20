#ifndef USI_OPTIONS_HPP
#define USI_OPTIONS_HPP

class USIOption {
public:
    int64_t byoyomi_margin;
    int64_t random_turn;
    int64_t USI_Hash;
    int64_t draw_turn;
    int64_t print_interval;
    int64_t thread_num;
    int64_t search_batch_size;
    int64_t search_limit;
    bool print_debug_info;
};

#endif