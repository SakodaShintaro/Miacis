﻿#ifndef MIACIS_SHOGI_HAND_HPP
#define MIACIS_SHOGI_HAND_HPP

#include "piece.hpp"

enum HandConst {
    //0000 0000 0000 0000 0000 0000 0011 1111 PAWN
    //0000 0000 0000 0000 0000 0011 1000 0000 LANCE
    //0000 0000 0000 0000 0011 1000 0000 0000 KNIGHT
    //0000 0000 0000 0011 1000 0000 0000 0000 SILVER
    //0000 0000 0011 1000 0000 0000 0000 0000 GOLD
    //0000 0001 1000 0000 0000 0000 0000 0000 BISHOP
    //0000 1100 0000 0000 0000 0000 0000 0000 ROOK

    // clang-format off
    HAND_PAWN_SHIFT   = 0,
    HAND_LANCE_SHIFT  = HAND_PAWN_SHIFT + 7,
    HAND_KNIGHT_SHIFT = HAND_LANCE_SHIFT + 4,
    HAND_SILVER_SHIFT = HAND_KNIGHT_SHIFT + 4,
    HAND_GOLD_SHIFT   = HAND_SILVER_SHIFT + 4,
    HAND_BISHOP_SHIFT = HAND_GOLD_SHIFT + 4,
    HAND_ROOK_SHIFT   = HAND_BISHOP_SHIFT + 3,

    HAND_PAWN_MASK   = 0b111111,
    HAND_LANCE_MASK  = 0b111 << HAND_LANCE_SHIFT,
    HAND_KNIGHT_MASK = 0b111 << HAND_KNIGHT_SHIFT,
    HAND_SILVER_MASK = 0b111 << HAND_SILVER_SHIFT,
    HAND_GOLD_MASK   = 0b111 << HAND_GOLD_SHIFT,
    HAND_BISHOP_MASK = 0b11 << HAND_BISHOP_SHIFT,
    HAND_ROOK_MASK   = 0b11 << HAND_ROOK_SHIFT,
    // clang-format on
};

// clang-format off
static int32_t PieceToHandShift[] = {
    0,
    HAND_PAWN_SHIFT,
    HAND_LANCE_SHIFT,
    HAND_KNIGHT_SHIFT,
    HAND_SILVER_SHIFT,
    HAND_GOLD_SHIFT,
    HAND_BISHOP_SHIFT,
    HAND_ROOK_SHIFT
};

static int32_t PieceToHandMask[] = {
    0,
    HAND_PAWN_MASK,
    HAND_LANCE_MASK,
    HAND_KNIGHT_MASK,
    HAND_SILVER_MASK,
    HAND_GOLD_MASK,
    HAND_BISHOP_MASK,
    HAND_ROOK_MASK,
};
// clang-format on

class Hand {
public:
    //コンストラクタ
    Hand() : hand_(0) {}

    //持ち駒の数を返す
    inline int32_t num(Piece p) const { return ((hand_ & PieceToHandMask[kind(p)]) >> PieceToHandShift[kind(p)]); }

    //capture(Piece型)を受け取って持ち駒を増減する
    inline void add(Piece p) { hand_ += 1 << PieceToHandShift[kind(p)]; }
    inline void sub(Piece p) { hand_ -= 1 << PieceToHandShift[kind(p)]; }

    //初期化のとき使う
    void set(Piece p, int32_t num) { hand_ += num << PieceToHandShift[kind(p)]; }

    //「lhsのどの種類の枚数もrhs以上であり、かつ少なくとも一種類はrhsより多い」かどうかを判定
    bool superior(const Hand rhs) const {
        bool over = false;
        for (Piece p : { PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK }) {
            if (num(p) < rhs.num(p)) {
                return false;
            } else if (num(p) > rhs.num(p)) {
                over = true;
            }
        }
        return over;
    }

    //superiorの逆
    bool inferior(const Hand rhs) const {
        bool under = false;
        for (Piece p : { PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK }) {
            if (num(p) > rhs.num(p)) {
                return false;
            } else if (num(p) < rhs.num(p)) {
                under = true;
            }
        }
        return under;
    }

    //zeroクリア
    void clear() { hand_ = 0; }

    //表示
    void print() const {
        for (Piece p = PAWN; p <= ROOK; p++)
            if (num(p)) {
                std::cout << PieceToStr[p] << num(p) << " ";
            }
        std::cout << std::endl;
    }

private:
    uint32_t hand_;
};

#endif //MIACIS_SHOGI_HAND_HPP