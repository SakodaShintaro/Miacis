#ifndef MIACIS_GO_SQUARE_HPP
#define MIACIS_GO_SQUARE_HPP

#include "piece.hpp"

namespace Go {

// clang-format off
enum Square {
    WALL00, WALL01, WALL02, WALL03, WALL04, WALL05, WALL06, WALL07, WALL08, WALL09,
    WALL10,   SQ11,   SQ12,   SQ13,   SQ14,   SQ15,   SQ16,   SQ17,   SQ18, WALL19,
    WALL20,   SQ21,   SQ22,   SQ23,   SQ24,   SQ25,   SQ26,   SQ27,   SQ28, WALL29,
    WALL30,   SQ31,   SQ32,   SQ33,   SQ34,   SQ35,   SQ36,   SQ37,   SQ38, WALL39,
    WALL40,   SQ41,   SQ42,   SQ43,   SQ44,   SQ45,   SQ46,   SQ47,   SQ48, WALL49,
    WALL50,   SQ51,   SQ52,   SQ53,   SQ54,   SQ55,   SQ56,   SQ57,   SQ58, WALL59,
    WALL60,   SQ61,   SQ62,   SQ63,   SQ64,   SQ65,   SQ66,   SQ67,   SQ68, WALL69,
    WALL70,   SQ71,   SQ72,   SQ73,   SQ74,   SQ75,   SQ76,   SQ77,   SQ78, WALL79,
    WALL80,   SQ81,   SQ82,   SQ83,   SQ84,   SQ85,   SQ86,   SQ87,   SQ88, WALL89,
    WALL90, WALL91, WALL92, WALL93, WALL94, WALL95, WALL96, WALL97, WALL98, WALL99,
    SquareNum,
};

enum File {
    File0, File1, File2, File3, File4, File5, File6, File7, File8, File9, FileNum,
};

enum Rank {
    Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9, RankNum,
};

// clang-format on

constexpr int32_t BOARD_WIDTH = 9;
constexpr int32_t SQUARE_NUM = BOARD_WIDTH * BOARD_WIDTH;

inline Square xy2square(int32_t x, int32_t y) { return Square(y * BOARD_WIDTH + x); }

enum Dir {
    H = 0,
    U = -1,       //上
    D = 1,        //下
    R = -RankNum, //右
    L = RankNum,  //左
    RU = R + U,   //右上
    RD = R + D,   //右下
    LD = L + D,   //左下
    LU = L + U,   //左上
};

inline Square operator+(Square sq, Dir diff) {
    return static_cast<Square>(static_cast<int32_t>(sq) + static_cast<int32_t>(diff));
}

inline int32_t operator<<(Square sq, int32_t shift) { return static_cast<int32_t>(static_cast<int32_t>(sq) << shift); }

extern const std::array<Square, SQUARE_NUM> SquareList;
extern const int32_t SquareToNum[];
extern const Dir DirList[8];

inline std::string fileToString(int32_t file) { return { (char)('H' - (BOARD_WIDTH - file)) }; }
extern const ArrayMap<std::string, RankNum> rankToString;

std::ostream& operator<<(std::ostream&, Square sq);

} // namespace Go

#endif //MIACIS_GO_SQUARE_HPP