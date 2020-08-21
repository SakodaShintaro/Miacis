#ifndef MIACIS_GO_SQUARE_HPP
#define MIACIS_GO_SQUARE_HPP

#include "piece.hpp"

namespace Go {

using Square = int32_t;

constexpr int32_t BOARD_WIDTH = 9;
constexpr int32_t SQUARE_NUM = BOARD_WIDTH * BOARD_WIDTH;

inline Square xy2square(int32_t x, int32_t y) { return Square(y * BOARD_WIDTH + x); }

enum Dir {
    H = 0,
    U = -1,           //上
    D = 1,            //下
    R = -BOARD_WIDTH, //右
    L = BOARD_WIDTH,  //左
    RU = R + U,       //右上
    RD = R + D,       //右下
    LD = L + D,       //左下
    LU = L + U,       //左上
};

inline Square operator+(Square sq, Dir diff) {
    return static_cast<Square>(static_cast<int32_t>(sq) + static_cast<int32_t>(diff));
}

extern const Dir DirList[8];

std::string squareToString(Square sq);

} // namespace Go

#endif //MIACIS_GO_SQUARE_HPP