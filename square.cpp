#include"square.hpp"
#include<array>

std::vector<Dir> CanMove[WHITE_ROOK_PROMOTE + 1];
Dir ConDirToOppositeDir[129];

void initCanMove() {
    CanMove[BLACK_PAWN] = { U };
    CanMove[BLACK_LANCE] = {};
    CanMove[BLACK_KNIGHT] = { RUU, LUU };
    CanMove[BLACK_SILVER] = { U, RU, RD, LD, LU };
    CanMove[BLACK_GOLD] = { U, RU, R, D, L, LU };
    CanMove[BLACK_BISHOP] = {};
    CanMove[BLACK_ROOK] = {};
    CanMove[BLACK_KING] = { U, RU, R, RD, D, LD, L, LU };
    CanMove[BLACK_PAWN_PROMOTE] = { U, RU, R, D, L, LU };
    CanMove[BLACK_LANCE_PROMOTE] = { U, RU, R, D, L, LU };
    CanMove[BLACK_KNIGHT_PROMOTE] = { U, RU, R, D, L, LU };
    CanMove[BLACK_SILVER_PROMOTE] = { U, RU, R, D, L, LU };
    CanMove[BLACK_BISHOP_PROMOTE] = { U, R, D, L };
    CanMove[BLACK_ROOK_PROMOTE] = { RU, RD, LD, LU };
    CanMove[WHITE_PAWN] = { D };
    CanMove[WHITE_LANCE] = {};
    CanMove[WHITE_KNIGHT] = { RDD, LDD };
    CanMove[WHITE_SILVER] = { RU, RD, D, LD, LU };
    CanMove[WHITE_GOLD] = { U, R, RD, D, LD, L };
    CanMove[WHITE_BISHOP] = {};
    CanMove[WHITE_ROOK] = {};
    CanMove[WHITE_KING] = { U, RU, R, RD, D, LD, L, LU };
    CanMove[WHITE_PAWN_PROMOTE] = { U, R, RD, D, LD, L };
    CanMove[WHITE_LANCE_PROMOTE] = { U, R, RD, D, LD, L };
    CanMove[WHITE_KNIGHT_PROMOTE] = { U, R, RD, D, LD, L };
    CanMove[WHITE_SILVER_PROMOTE] = { U, R, RD, D, LD, L };
    CanMove[WHITE_BISHOP_PROMOTE] = { U, R, D, L };
    CanMove[WHITE_ROOK_PROMOTE] = { RU, RD, LD, LU };
}

std::vector<Dir> CanJump[WHITE_ROOK_PROMOTE + 1];
void initCanJump() {
    CanJump[BLACK_PAWN] = {};
    CanJump[BLACK_LANCE] = { U };
    CanJump[BLACK_KNIGHT] = {};
    CanJump[BLACK_SILVER] = {};
    CanJump[BLACK_GOLD] = {};
    CanJump[BLACK_BISHOP] = { RU, RD, LD, LU };
    CanJump[BLACK_ROOK] = { U, R, D, L };
    CanJump[BLACK_PAWN_PROMOTE] = {};
    CanJump[BLACK_LANCE_PROMOTE] = {};
    CanJump[BLACK_KNIGHT_PROMOTE] = {};
    CanJump[BLACK_SILVER_PROMOTE] = {};
    CanJump[BLACK_BISHOP_PROMOTE] = { RU, RD, LD, LU };
    CanJump[BLACK_ROOK_PROMOTE] = { U, R, D, L };
    CanJump[WHITE_PAWN] = {};
    CanJump[WHITE_LANCE] = { D };
    CanJump[WHITE_KNIGHT] = {};
    CanJump[WHITE_SILVER] = {};
    CanJump[WHITE_GOLD] = {};
    CanJump[WHITE_BISHOP] = { RU, RD, LD, LU };
    CanJump[WHITE_ROOK] = { U, R, D, L };
    CanJump[WHITE_PAWN_PROMOTE] = {};
    CanJump[WHITE_LANCE_PROMOTE] = {};
    CanJump[WHITE_KNIGHT_PROMOTE] = {};
    CanJump[WHITE_SILVER_PROMOTE] = {};
    CanJump[WHITE_BISHOP_PROMOTE] = { RU, RD, LD, LU };
    CanJump[WHITE_ROOK_PROMOTE] = { U, R, D, L };
}

const std::array<Square, 81> SquareList = {
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

const int SquareToNum[] = {
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

std::ostream& operator<<(std::ostream& os, Square sq) {
    os << SquareToFile[sq] << SquareToRank[sq];
    return os;
}

int32_t mirrorSqNum(int32_t sq_num) {
    const int32_t f = sq_num / 9 + 1;
    const int32_t r = sq_num % 9 + 1;
    const Square mirror = FRToSquare[File9 - f + 1][r];
    return SquareToNum[mirror];
}

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