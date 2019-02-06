#pragma once
#ifndef USI_OPTIONS_HPP
#define USI_OPTIONS_HPP

class USIOption{
public:
	int64_t byoyomi_margin;
	int64_t random_turn;
    int64_t USI_Hash;
    int64_t thread_num;
    int64_t draw_turn;
    int64_t draw_score;

    //探索中に参照するものら
	int64_t limit_msec;
	bool stop_signal;
	bool print_usi_info;

    int64_t playout_limit;
};

extern USIOption usi_option;

#endif