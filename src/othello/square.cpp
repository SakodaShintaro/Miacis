#include "square.hpp"

namespace Othello {

// clang-format off
const std::array<Square, SQUARE_NUM> SquareList = {
    SQ11, SQ12, SQ13, SQ14, SQ15, SQ16, SQ17, SQ18,
    SQ21, SQ22, SQ23, SQ24, SQ25, SQ26, SQ27, SQ28,
    SQ31, SQ32, SQ33, SQ34, SQ35, SQ36, SQ37, SQ38,
    SQ41, SQ42, SQ43, SQ44, SQ45, SQ46, SQ47, SQ48,
    SQ51, SQ52, SQ53, SQ54, SQ55, SQ56, SQ57, SQ58,
    SQ61, SQ62, SQ63, SQ64, SQ65, SQ66, SQ67, SQ68,
    SQ71, SQ72, SQ73, SQ74, SQ75, SQ76, SQ77, SQ78,
    SQ81, SQ82, SQ83, SQ84, SQ85, SQ86, SQ87, SQ88,
};

const int32_t SquareToNum[] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1,  0,  1,  2,  3,  4,  5,  6,  7, -1,
    -1,  8,  9, 10, 11, 12, 13, 14, 15, -1,
    -1, 16, 17, 18, 19, 20, 21, 22, 23, -1,
    -1, 24, 25, 26, 27, 28, 29, 30, 31, -1,
    -1, 32, 33, 34, 35, 36, 37, 38, 39, -1,
    -1, 40, 41, 42, 43, 44, 45, 46, 47, -1,
    -1, 48, 49, 50, 51, 52, 53, 54, 55, -1,
    -1, 56, 57, 58, 59, 60, 61, 62, 63, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
};
// clang-format on

std::ostream& operator<<(std::ostream& os, Square sq) {
    os << fileToString[SquareToFile[sq]] << rankToString[SquareToRank[sq]];
    return os;
}

const ArrayMap<std::string, FileNum> fileToString({
    { File0, "P" },
    { File1, "H" },
    { File2, "G" },
    { File3, "F" },
    { File4, "E" },
    { File5, "D" },
    { File6, "C" },
    { File7, "B" },
    { File8, "A" },
});

const ArrayMap<std::string, RankNum> rankToString({
    { Rank0, "A" },
    { Rank1, "1" },
    { Rank2, "2" },
    { Rank3, "3" },
    { Rank4, "4" },
    { Rank5, "5" },
    { Rank6, "6" },
    { Rank7, "7" },
    { Rank8, "8" },
});

const Dir DirList[8] = {
    //前から時計回りに
    U, RU, R, RD, D, LD, L, LU
};

} // namespace Othello