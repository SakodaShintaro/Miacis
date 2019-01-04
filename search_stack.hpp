#pragma once
#ifndef SEARCH_STACK_HPP
#define SEARCH_STACK_HPP

#ifdef USE_SEARCH_STACK

#include"move.hpp"
#include"position.hpp"
#include"history.hpp"

struct SearchStack {
    Move killers[2];
    bool can_null_move;

    void updateKillers(const Move& move) {
        //参考:https://qiita.com/ak11/items/0c1d20753b1073788275
        if (killers[0] != move) {
            killers[1] = killers[0];
            killers[0] = move;
        }
    }
};

#endif // !USE_SEARCH_STACK

#endif // !SEARCH_STACK_HPP