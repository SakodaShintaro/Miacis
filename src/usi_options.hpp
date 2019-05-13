#ifndef USI_OPTIONS_HPP
#define USI_OPTIONS_HPP

class USIOption{
public:
	int64_t byoyomi_margin;
	int64_t random_turn;
    int64_t USI_Hash;
    int64_t draw_turn;
    int64_t print_interval;
	uint64_t thread_num;
	uint64_t search_batch_size;
	bool print_debug_info;

	//探索中に参照するものら
	int64_t limit_msec;
	bool stop_signal;

    int64_t search_limit;
};

extern USIOption usi_option;

#endif