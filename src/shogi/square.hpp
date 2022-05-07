﻿#ifndef MIACIS_SHOGI_SQUARE_HPP
#define MIACIS_SHOGI_SQUARE_HPP

#include "piece.hpp"

// clang-format off
enum Square {
    WALL00, WALL01, WALL02, WALL03, WALL04, WALL05, WALL06, WALL07, WALL08, WALL09, WALL0A,
    WALL10,   SQ11,   SQ12,   SQ13,   SQ14,   SQ15,   SQ16,   SQ17,   SQ18,   SQ19, WALL1A,
    WALL20,   SQ21,   SQ22,   SQ23,   SQ24,   SQ25,   SQ26,   SQ27,   SQ28,   SQ29, WALL2A,
    WALL30,   SQ31,   SQ32,   SQ33,   SQ34,   SQ35,   SQ36,   SQ37,   SQ38,   SQ39, WALL3A,
    WALL40,   SQ41,   SQ42,   SQ43,   SQ44,   SQ45,   SQ46,   SQ47,   SQ48,   SQ49, WALL4A,
    WALL50,   SQ51,   SQ52,   SQ53,   SQ54,   SQ55,   SQ56,   SQ57,   SQ58,   SQ59, WALL5A,
    WALL60,   SQ61,   SQ62,   SQ63,   SQ64,   SQ65,   SQ66,   SQ67,   SQ68,   SQ69, WALL6A,
    WALL70,   SQ71,   SQ72,   SQ73,   SQ74,   SQ75,   SQ76,   SQ77,   SQ78,   SQ79, WALL7A,
    WALL80,   SQ81,   SQ82,   SQ83,   SQ84,   SQ85,   SQ86,   SQ87,   SQ88,   SQ89, WALL8A,
    WALL90,   SQ91,   SQ92,   SQ93,   SQ94,   SQ95,   SQ96,   SQ97,   SQ98,   SQ99, WALL9A,
    WALLA0, WALLA1, WALLA2, WALLA3, WALLA4, WALLA5, WALLA6, WALLA7, WALLA8, WALLA9, WALLAA,
    SquareNum,
};

enum File {
    File0, File1, File2, File3, File4, File5, File6, File7, File8, File9, FileA, FileNum,
};

enum Rank {
    Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9, RankA, RankNum,
};

enum DiagR {
    DiagR0, DiagR1, DiagR2, DiagR3, DiagR4, DiagR5, DiagR6, DiagR7, DiagR8, DiagR9, DiagRA, DiagRB, DiagRC, DiagRD, DiagRE, DiagRF, DiagRG, DiagRH,
};

enum DiagL {
    DiagL0, DiagL1, DiagL2, DiagL3, DiagL4, DiagL5, DiagL6, DiagL7, DiagL8, DiagL9, DiagLA, DiagLB, DiagLC, DiagLD, DiagLE, DiagLF, DiagLG, DiagLH,
};

const Rank SquareToRank[SquareNum] = {
    Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9, RankA,
    Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9, RankA,
    Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9, RankA,
    Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9, RankA,
    Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9, RankA,
    Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9, RankA,
    Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9, RankA,
    Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9, RankA,
    Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9, RankA,
    Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9, RankA,
    Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9, RankA,
};

const File SquareToFile[SquareNum] = {
    File0, File0, File0, File0, File0, File0, File0, File0, File0, File0, File0,
    File1, File1, File1, File1, File1, File1, File1, File1, File1, File1, File1,
    File2, File2, File2, File2, File2, File2, File2, File2, File2, File2, File2,
    File3, File3, File3, File3, File3, File3, File3, File3, File3, File3, File3,
    File4, File4, File4, File4, File4, File4, File4, File4, File4, File4, File4,
    File5, File5, File5, File5, File5, File5, File5, File5, File5, File5, File5,
    File6, File6, File6, File6, File6, File6, File6, File6, File6, File6, File6,
    File7, File7, File7, File7, File7, File7, File7, File7, File7, File7, File7,
    File8, File8, File8, File8, File8, File8, File8, File8, File8, File8, File8,
    File9, File9, File9, File9, File9, File9, File9, File9, File9, File9, File9,
    FileA, FileA, FileA, FileA, FileA, FileA, FileA, FileA, FileA, FileA, FileA,
};

//斜め方向右上がり
const DiagR SquareToDiagR[SquareNum] = {
    DiagR0, DiagR0, DiagR0, DiagR0, DiagR0, DiagR0, DiagR0, DiagR0, DiagR0, DiagR0, DiagR0,
    DiagR0, DiagR1, DiagR2, DiagR3, DiagR4, DiagR5, DiagR6, DiagR7, DiagR8, DiagR9, DiagR0,
    DiagR0, DiagR2, DiagR3, DiagR4, DiagR5, DiagR6, DiagR7, DiagR8, DiagR9, DiagRA, DiagR0,
    DiagR0, DiagR3, DiagR4, DiagR5, DiagR6, DiagR7, DiagR8, DiagR9, DiagRA, DiagRB, DiagR0,
    DiagR0, DiagR4, DiagR5, DiagR6, DiagR7, DiagR8, DiagR9, DiagRA, DiagRB, DiagRC, DiagR0,
    DiagR0, DiagR5, DiagR6, DiagR7, DiagR8, DiagR9, DiagRA, DiagRB, DiagRC, DiagRD, DiagR0,
    DiagR0, DiagR6, DiagR7, DiagR8, DiagR9, DiagRA, DiagRB, DiagRC, DiagRD, DiagRE, DiagR0,
    DiagR0, DiagR7, DiagR8, DiagR9, DiagRA, DiagRB, DiagRC, DiagRD, DiagRE, DiagRF, DiagR0,
    DiagR0, DiagR8, DiagR9, DiagRA, DiagRB, DiagRC, DiagRD, DiagRE, DiagRF, DiagRG, DiagR0,
    DiagR0, DiagR9, DiagRA, DiagRB, DiagRC, DiagRD, DiagRE, DiagRF, DiagRG, DiagRH, DiagR0,
    DiagR0, DiagR0, DiagR0, DiagR0, DiagR0, DiagR0, DiagR0, DiagR0, DiagR0, DiagR0, DiagR0,
};

//斜め方向左上がり
const DiagL SquareToDiagL[SquareNum] = {
    DiagL0, DiagL0, DiagL0, DiagL0, DiagL0, DiagL0, DiagL0, DiagL0, DiagL0, DiagL0, DiagL0,
    DiagL0, DiagL9, DiagL8, DiagL7, DiagL6, DiagL5, DiagL4, DiagL3, DiagL2, DiagL1, DiagL0,
    DiagL0, DiagLA, DiagL9, DiagL8, DiagL7, DiagL6, DiagL5, DiagL4, DiagL3, DiagL2, DiagL0,
    DiagL0, DiagLB, DiagLA, DiagL9, DiagL8, DiagL7, DiagL6, DiagL5, DiagL4, DiagL3, DiagL0,
    DiagL0, DiagLC, DiagLB, DiagLA, DiagL9, DiagL8, DiagL7, DiagL6, DiagL5, DiagL4, DiagL0,
    DiagL0, DiagLD, DiagLC, DiagLB, DiagLA, DiagL9, DiagL8, DiagL7, DiagL6, DiagL5, DiagL0,
    DiagL0, DiagLE, DiagLD, DiagLC, DiagLB, DiagLA, DiagL9, DiagL8, DiagL7, DiagL6, DiagL0,
    DiagL0, DiagLF, DiagLE, DiagLD, DiagLC, DiagLB, DiagLA, DiagL9, DiagL8, DiagL7, DiagL0,
    DiagL0, DiagLG, DiagLF, DiagLE, DiagLD, DiagLC, DiagLB, DiagLA, DiagL9, DiagL8, DiagL0,
    DiagL0, DiagLH, DiagLG, DiagLF, DiagLE, DiagLD, DiagLC, DiagLB, DiagLA, DiagL9, DiagL0,
    DiagL0, DiagL0, DiagL0, DiagL0, DiagL0, DiagL0, DiagL0, DiagL0, DiagL0, DiagL0, DiagL0,
};

const Square FRToSquare[FileNum][RankNum] = {
    {WALL00, WALL01, WALL02, WALL03, WALL04, WALL05, WALL06, WALL07, WALL08, WALL09, WALL0A},
    {WALL10, SQ11,   SQ12,   SQ13,   SQ14,   SQ15,   SQ16,   SQ17,   SQ18,   SQ19,   WALL1A},
    {WALL20, SQ21,   SQ22,   SQ23,   SQ24,   SQ25,   SQ26,   SQ27,   SQ28,   SQ29,   WALL2A},
    {WALL30, SQ31,   SQ32,   SQ33,   SQ34,   SQ35,   SQ36,   SQ37,   SQ38,   SQ39,   WALL3A},
    {WALL40, SQ41,   SQ42,   SQ43,   SQ44,   SQ45,   SQ46,   SQ47,   SQ48,   SQ49,   WALL4A},
    {WALL50, SQ51,   SQ52,   SQ53,   SQ54,   SQ55,   SQ56,   SQ57,   SQ58,   SQ59,   WALL5A},
    {WALL60, SQ61,   SQ62,   SQ63,   SQ64,   SQ65,   SQ66,   SQ67,   SQ68,   SQ69,   WALL6A},
    {WALL70, SQ71,   SQ72,   SQ73,   SQ74,   SQ75,   SQ76,   SQ77,   SQ78,   SQ79,   WALL7A},
    {WALL80, SQ81,   SQ82,   SQ83,   SQ84,   SQ85,   SQ86,   SQ87,   SQ88,   SQ89,   WALL8A},
    {WALL90, SQ91,   SQ92,   SQ93,   SQ94,   SQ95,   SQ96,   SQ97,   SQ98,   SQ99,   WALL9A},
    {WALLA0, WALLA1, WALLA2, WALLA3, WALLA4, WALLA5, WALLA6, WALLA7, WALLA8, WALLA9, WALLAA},
};
// clang-format on

constexpr int64_t BOARD_WIDTH = 9;
constexpr int64_t SQUARE_NUM = BOARD_WIDTH * BOARD_WIDTH;

enum Dir {
    H = 0,
    U = -1,       //上
    D = 1,        //下
    R = -11,      //右
    L = 11,       //左
    RU = R + U,   //右上
    RD = R + D,   //右下
    LD = L + D,   //左下
    LU = L + U,   //左上
    RUU = RU + U, //右上上
    RDD = RD + D, //右下下
    LDD = LD + D, //左下下
    LUU = LU + U, //左上上
};

inline bool isOnBoard(Square pos) {
    return (Rank1 <= SquareToRank[pos] && SquareToRank[pos] <= Rank9 && File1 <= SquareToFile[pos] && SquareToFile[pos] <= File9);
}

inline Dir directionAtoB(Square A, Square B) {
    //8方向のうちどれかか、あるいはどれでもないかだけ判定できればいい
    //Aの位置を0とすると周囲8マスは
    //10 -1 -12
    //11  0 -11
    //12  1 -10
    //だから差の正負と段、筋、斜めの一致具合で方向がわかるはず
    if (A == B) return H;
    else if (B - A > 0) {
        if (SquareToRank[A] == SquareToRank[B]) return L;
        if (SquareToFile[A] == SquareToFile[B]) return D;
        if (SquareToDiagR[A] == SquareToDiagR[B]) return LU;
        if (SquareToDiagL[A] == SquareToDiagL[B]) return LD;
    } else {
        if (SquareToRank[A] == SquareToRank[B]) return R;
        if (SquareToFile[A] == SquareToFile[B]) return U;
        if (SquareToDiagR[A] == SquareToDiagR[B]) return RD;
        if (SquareToDiagL[A] == SquareToDiagL[B]) return RU;
    }
    return H;
}

inline Square operator+(Square sq, Dir diff) { return (Square)((int32_t)sq + (int32_t)diff); }
inline int32_t operator<<(Square sq, int32_t shift) { return (int32_t)sq << shift; }

extern const std::array<Square, SQUARE_NUM> SquareList;
extern const int32_t SquareToNum[];
extern const Square InvSquare[];

extern const ArrayMap<std::string, FileNum> fileToString;
extern const ArrayMap<std::string, RankNum> rankToString;

std::ostream& operator<<(std::ostream&, Square sq);

#endif //MIACIS_SHOGI_SQUARE_HPP