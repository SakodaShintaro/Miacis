#pragma once
#ifndef USI_OPTIONS_HPP
#define USI_OPTIONS_HPP

#include"MCTSearcher.hpp"

class USIOption{
public:
	int64_t byoyomi_margin;
	int64_t random_turn;
    int64_t USI_Hash;
    int64_t thread_num;
    int64_t draw_turn;
    int64_t draw_score;

#ifdef USE_MCTS
    int64_t playout_limit;
#else
    Depth depth_limit;
#endif
};

extern USIOption usi_option;

#endif