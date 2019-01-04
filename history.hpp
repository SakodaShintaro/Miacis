#pragma once
#ifndef HISTORY_HPP
#define HISTORY_HPP

#include"piece.hpp"
#include"square.hpp"
#include"move.hpp"
#include<queue>

class History {
private:
    Score table_[PieceNum][SquareNum][2]; //4 * 88 * 121 * 2 = 85KB
public:
    void updateBetaCutMove(const Move move, const Depth depth) {
        int d = depth / PLY;
        (*this)[move] += (d > 17 ? 0 : d * d + 2 * d - 2);
    }

    void updateNonBetaCutMove(const Move move, const Depth depth) {
        int d = depth / PLY;
        (*this)[move] -= (d > 17 ? 0 : d * d + 2 * d - 2);
    }

    Score operator[](const Move& move) const {
        return table_[move.subject()][move.to()][move.capture() != EMPTY];
    }
    Score& operator[](const Move& move) {
        return table_[move.subject()][move.to()][move.capture() != EMPTY];
    }

    void clear() {
        for (int piece = 0; piece < PieceNum; ++piece) {
            for (int to = 0; to < SquareNum; ++to) {
                table_[piece][to][0] = SCORE_ZERO;
                table_[piece][to][1] = SCORE_ZERO;
            }
        }
    }
};

#ifdef USE_MOVEHISTORY

class MoveHistory {
public:
    Move operator[](const Move& move) const {
        return table_[move.subject()][move.to()];
    }
    void update(Move pre, Move beta_cut) {
        table_[pre.subject()][pre.to()] = beta_cut;
    }
private:
    Move table_[PieceNum][SquareNum];
};

#endif

#endif // !HISTORY_HPP