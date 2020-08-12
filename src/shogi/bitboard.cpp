#include "bitboard.hpp"

Bitboard BOARD_BB;
Bitboard SQUARE_BB[SquareNum];
Bitboard FILE_BB[FileNum];
Bitboard RANK_BB[RankNum];
Bitboard PROMOTION_ZONE_BB[ColorNum];
Bitboard FRONT_BB[ColorNum][RankNum];
Bitboard BETWEEN_BB[SquareNum][SquareNum];

Bitboard PAWN_CONTROL_BB[ColorNum][SquareNum];
Bitboard KNIGHT_CONTROL_BB[ColorNum][SquareNum];
Bitboard SILVER_CONTROL_BB[ColorNum][SquareNum];
Bitboard GOLD_CONTROL_BB[ColorNum][SquareNum];

Bitboard BishopEffect[2][SquareNum][128];
Bitboard BishopEffectMask[2][SquareNum];

uint64_t RookFileEffect[RankNum][128];
Bitboard RookRankEffect[FileNum][128];

Bitboard KING_CONTROL_BB[SquareNum];

// clang-format off
int32_t Slide[] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1,
    -1, 10, 10, 10, 10, 10, 10, 10, 10, 10, -1,
    -1, 19, 19, 19, 19, 19, 19, 19, 19, 19, -1,
    -1, 28, 28, 28, 28, 28, 28, 28, 28, 28, -1,
    -1, 37, 37, 37, 37, 37, 37, 37, 37, 37, -1,
    -1, 46, 46, 46, 46, 46, 46, 46, 46, 46, -1,
    -1, 55, 55, 55, 55, 55, 55, 55, 55, 55, -1,
    -1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1,
    -1, 10, 10, 10, 10, 10, 10, 10, 10, 10, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
};
// clang-format on

std::ostream& operator<<(std::ostream& os, const Bitboard& rhs) {
    for (int r = Rank1; r <= Rank9; ++r) {
        for (int f = File9; f >= File1; --f) {
            Bitboard target = rhs & SQUARE_BB[FRToSquare[f][r]];
            os << ((bool)target ? " *" : " .");
        }
        os << std::endl;
    }
    return os;
}

void Bitboard::init() {
    //1.SQUARE_BB
    for (auto sq : SquareList) {
        SQUARE_BB[sq] = Bitboard(sq);
        BOARD_BB |= SQUARE_BB[sq];
    }

    //2.FILE_BB,RANK_BB
    for (int f = File1; f <= File9; ++f) {
        for (int r = Rank1; r <= Rank9; ++r) {
            FILE_BB[f] |= SQUARE_BB[FRToSquare[f][r]];
            RANK_BB[r] |= SQUARE_BB[FRToSquare[f][r]];
        }
    }

    //3.PROMOTION_ZONE_BBとFRONT_BB
    PROMOTION_ZONE_BB[BLACK] = RANK_BB[Rank1] | RANK_BB[Rank2] | RANK_BB[Rank3];
    PROMOTION_ZONE_BB[WHITE] = RANK_BB[Rank7] | RANK_BB[Rank8] | RANK_BB[Rank9];

    for (int rank = Rank1; rank <= Rank9; ++rank) {
        for (int black_front = rank - 1; black_front >= Rank1; --black_front) {
            FRONT_BB[BLACK][rank] |= RANK_BB[black_front];
        }
        for (int white_front = rank + 1; white_front <= Rank9; ++white_front) {
            FRONT_BB[WHITE][rank] |= RANK_BB[white_front];
        }
    }

    //BETWEEN_BB
    for (Square sq1 : SquareList) {
        for (Square sq2 : SquareList) {
            auto dir = directionAtoB(sq1, sq2);
            if (dir == H) {
                continue;
            }
            //1マスずつたどっていく
            for (Square between = sq1 + dir; between != sq2; between = between + dir) {
                BETWEEN_BB[sq1][sq2] |= SQUARE_BB[between];
            }
        }
    }

    //4.飛び利き
    auto indexToOccupied = [](const int index, const int bits, Bitboard mask) {
        Bitboard result = Bitboard(0, 0);
        for (int i = 0; i < bits; ++i) {
            const Square sq = mask.pop();
            if (index & (1u << i)) {
                result |= SQUARE_BB[sq];
            }
        }
        return result;
    };

    //角の利きのために用意しておく
    //n = 0が右上・左下
    //n = 1が左上・右下
    static const Dir diagonal_deltas[2][2] = { { RU, LD }, { RD, LU } };

    auto calcBishopEffectMask = [](Square sq, int n) {
        auto result = Bitboard(0, 0);

        for (auto delta : diagonal_deltas[n]) {
            for (Square to = sq + delta; isOnBoard(to); to = to + delta) {
                result |= SQUARE_BB[to];
            }
        }

        //端は関係ないので外す
        result = result & ~(FILE_BB[File1]);
        result = result & ~(FILE_BB[File9]);
        result = result & ~(RANK_BB[Rank1]);
        result = result & ~(RANK_BB[Rank9]);
        return result;
    };

    //角の利きを初期化
    for (int n : { 0, 1 }) {
        for (auto sq : SquareList) {
            Bitboard& mask = BishopEffectMask[n][sq];
            mask = calcBishopEffectMask(sq, n);

            assert(!mask.crossOver());

            //全てのBitが立っている場合が最大
            auto bits = (int)(mask.pop_count());
            const int num = (1u << bits);
            for (int i = 0; i < num; ++i) {
                //邪魔駒の位置を示すindexであるiからoccupiedへ変換する
                Bitboard occupied = indexToOccupied(i, bits, mask);
                uint64_t index = occupiedToIndex(occupied, BishopEffectMask[n][sq]);

                //occupiedを考慮した利きを求める
                for (auto delta : diagonal_deltas[n]) {
                    for (Square to = sq + delta; isOnBoard(to); to = to + delta) {
                        BishopEffect[n][sq][index] |= SQUARE_BB[to];

                        //邪魔駒があったらそこまで
                        if (occupied & SQUARE_BB[to]) {
                            break;
                        }
                    }
                }
            }
        }
    }

    //飛車の縦方向
    for (int rank = Rank1; rank <= Rank9; ++rank) {
        const int num1s = 7;
        for (uint64_t i = 0; i < (1u << num1s); ++i) {
            //iが邪魔駒の配置を表したindex
            //1つシフトすればそのまま2~8段目のマスの邪魔駒を表す
            int occupied = (i << 1);
            uint64_t bb = 0;

            //上に利きを伸ばす
            for (int r = rank - 1; r >= Rank1; --r) {
                bb |= (1ULL << SquareToNum[FRToSquare[File1][r]]);
                //邪魔駒があったらそこまで
                if (occupied & (1 << (r - Rank1))) {
                    break;
                }
            }

            //下に利きを伸ばす
            for (int r = rank + 1; r <= Rank9; ++r) {
                bb |= (1LL << SquareToNum[FRToSquare[File1][r]]);
                //邪魔駒があったらそこまで
                if (occupied & (1 << (r - Rank1))) {
                    break;
                }
            }
            RookFileEffect[rank][i] = bb;
        }
    }

    //飛車の横方向
    for (int file = File1; file <= File9; ++file) {
        const int num1s = 7;
        for (int i = 0; i < (1 << num1s); ++i) {
            int j = i << 1;
            Bitboard bb(0, 0);
            for (int f = file - 1; f >= File1; --f) {
                bb |= SQUARE_BB[FRToSquare[f][Rank1]];
                if (j & (1 << (f - File1))) {
                    break;
                }
            }
            for (int f = file + 1; f <= File9; ++f) {
                bb |= SQUARE_BB[FRToSquare[f][Rank1]];
                if (j & (1u << (f - File1))) {
                    break;
                }
            }
            RookRankEffect[file][i] = bb;
        }
    }

    //5.近接駒の利き
    for (auto sq : SquareList) {
        //歩
        if (isOnBoard(sq + U)) {
            PAWN_CONTROL_BB[BLACK][sq] |= SQUARE_BB[sq + U];
        }
        if (isOnBoard(sq + D)) {
            PAWN_CONTROL_BB[WHITE][sq] |= SQUARE_BB[sq + D];
        }

        //香車は飛車の利きを用いて計算するのでテーブルは必要ない

        //桂馬
        if (SquareToRank[sq] >= Rank3) {
            if (isOnBoard(sq + RUU)) {
                KNIGHT_CONTROL_BB[BLACK][sq] |= SQUARE_BB[sq + RUU];
            }
            if (isOnBoard(sq + LUU)) {
                KNIGHT_CONTROL_BB[BLACK][sq] |= SQUARE_BB[sq + LUU];
            }
        }
        if (SquareToRank[sq] <= Rank7) {
            if (isOnBoard(sq + RDD)) {
                KNIGHT_CONTROL_BB[WHITE][sq] |= SQUARE_BB[sq + RDD];
            }
            if (isOnBoard(sq + LDD)) {
                KNIGHT_CONTROL_BB[WHITE][sq] |= SQUARE_BB[sq + LDD];
            }
        }

        //銀
        //先手の銀
        for (Dir delta : { U, RU, RD, LD, LU }) {
            Square to = sq + delta;
            if (isOnBoard(to)) {
                SILVER_CONTROL_BB[BLACK][sq] |= SQUARE_BB[to];
            }
        }
        //後手の銀
        for (Dir delta : { RU, RD, D, LD, LU }) {
            Square to = sq + delta;
            if (isOnBoard(to)) {
                SILVER_CONTROL_BB[WHITE][sq] |= SQUARE_BB[to];
            }
        }

        //金
        for (Dir delta : { U, RU, R, D, L, LU }) {
            Square to = sq + delta;
            if (isOnBoard(to)) {
                GOLD_CONTROL_BB[BLACK][sq] |= SQUARE_BB[to];
            }
        }
        for (Dir delta : { U, R, RD, D, LD, L }) {
            Square to = sq + delta;
            if (isOnBoard(to)) {
                GOLD_CONTROL_BB[WHITE][sq] |= SQUARE_BB[to];
            }
        }

        //飛車・角は前ステップで初期化した

        //王
        KING_CONTROL_BB[sq] = bishopControl(sq, BOARD_BB) | rookControl(sq, BOARD_BB);

        //と金・成香・成桂・成銀は金と同じ動き
        //竜・馬は飛車角と王を合成すればいい
    }
}

Bitboard::Bitboard(Square sq) {
    if (sq <= SQ79) {
        board_[0] = 1ULL << SquareToNum[sq];
        board_[1] = 0;
    } else {
        board_[0] = 0;
        board_[1] = 1ULL << (SquareToNum[sq] - SquareToNum[SQ81]);
    }
}