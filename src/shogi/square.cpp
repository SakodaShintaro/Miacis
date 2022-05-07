#include "square.hpp"

// clang-format off
const std::array<Square, SQUARE_NUM> SquareList = {
    SQ11, SQ12, SQ13, SQ14, SQ15, SQ16, SQ17, SQ18, SQ19,
    SQ21, SQ22, SQ23, SQ24, SQ25, SQ26, SQ27, SQ28, SQ29,
    SQ31, SQ32, SQ33, SQ34, SQ35, SQ36, SQ37, SQ38, SQ39,
    SQ41, SQ42, SQ43, SQ44, SQ45, SQ46, SQ47, SQ48, SQ49,
    SQ51, SQ52, SQ53, SQ54, SQ55, SQ56, SQ57, SQ58, SQ59,
    SQ61, SQ62, SQ63, SQ64, SQ65, SQ66, SQ67, SQ68, SQ69,
    SQ71, SQ72, SQ73, SQ74, SQ75, SQ76, SQ77, SQ78, SQ79,
    SQ81, SQ82, SQ83, SQ84, SQ85, SQ86, SQ87, SQ88, SQ89,
    SQ91, SQ92, SQ93, SQ94, SQ95, SQ96, SQ97, SQ98, SQ99
};

const int32_t SquareToNum[] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1,  0,  1,  2,  3,  4,  5,  6,  7,  8, -1,
    -1,  9, 10, 11, 12, 13, 14, 15, 16, 17, -1,
    -1, 18, 19, 20, 21, 22, 23, 24, 25, 26, -1,
    -1, 27, 28, 29, 30, 31, 32, 33, 34, 35, -1,
    -1, 36, 37, 38, 39, 40, 41, 42, 43, 44, -1,
    -1, 45, 46, 47, 48, 49, 50, 51, 52, 53, -1,
    -1, 54, 55, 56, 57, 58, 59, 60, 61, 62, -1,
    -1, 63, 64, 65, 66, 67, 68, 69, 70, 71, -1,
    -1, 72, 73, 74, 75, 76, 77, 78, 79, 80, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
};

const Square InvSquare[] = {
    WALL00, WALL01, WALL02, WALL03, WALL04, WALL05, WALL06, WALL07, WALL08, WALL09, WALL0A,
    WALL10, SQ99,   SQ98,   SQ97,   SQ96,   SQ95,   SQ94,   SQ93,   SQ92,   SQ91,   WALL1A,
    WALL20, SQ89,   SQ88,   SQ87,   SQ86,   SQ85,   SQ84,   SQ83,   SQ82,   SQ81,   WALL2A,
    WALL30, SQ79,   SQ78,   SQ77,   SQ76,   SQ75,   SQ74,   SQ73,   SQ72,   SQ71,   WALL3A,
    WALL40, SQ69,   SQ68,   SQ67,   SQ66,   SQ65,   SQ64,   SQ63,   SQ62,   SQ61,   WALL4A,
    WALL50, SQ59,   SQ58,   SQ57,   SQ56,   SQ55,   SQ54,   SQ53,   SQ52,   SQ51,   WALL5A,
    WALL60, SQ49,   SQ48,   SQ47,   SQ46,   SQ45,   SQ44,   SQ43,   SQ42,   SQ41,   WALL6A,
    WALL70, SQ39,   SQ38,   SQ37,   SQ36,   SQ35,   SQ34,   SQ33,   SQ32,   SQ31,   WALL7A,
    WALL80, SQ29,   SQ28,   SQ27,   SQ26,   SQ25,   SQ24,   SQ23,   SQ22,   SQ21,   WALL8A,
    WALL90, SQ19,   SQ18,   SQ17,   SQ16,   SQ15,   SQ14,   SQ13,   SQ12,   SQ11,   WALL9A,
    WALLA0, WALLA1, WALLA2, WALLA3, WALLA4, WALLA5, WALLA6, WALLA7, WALLA8, WALLA9, WALLAA,
};

const ArrayMap<std::string, FileNum> fileToString({
    { File1, "１" },
    { File2, "２" },
    { File3, "３" },
    { File4, "４" },
    { File5, "５" },
    { File6, "６" },
    { File7, "７" },
    { File8, "８" },
    { File9, "９" }
});

const ArrayMap<std::string, RankNum> rankToString({
    { Rank1, "一" },
    { Rank2, "二" },
    { Rank3, "三" },
    { Rank4, "四" },
    { Rank5, "五" },
    { Rank6, "六" },
    { Rank7, "七" },
    { Rank8, "八" },
    { Rank9, "九" }
});
// clang-format on

std::ostream& operator<<(std::ostream& os, Square sq) {
    os << SquareToFile[sq] << SquareToRank[sq];
    return os;
}
