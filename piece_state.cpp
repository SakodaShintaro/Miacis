#include"piece_state.hpp"
#include"square.hpp"

PieceState PieceToStateIndex[PieceNum];
PieceState invPieceStateIndex[PieceStateNum];

//PieceStateIndex[piece] = そのpieceのSQ11におけるPieceStateが返るように初期化
void initPieceToStateIndex() {
    PieceToStateIndex[PAWN] = black_hand_pawn;
    PieceToStateIndex[LANCE] = black_hand_lance;
    PieceToStateIndex[KNIGHT] = black_hand_knight;
    PieceToStateIndex[SILVER] = black_hand_silver;
    PieceToStateIndex[GOLD] = black_hand_gold;
    PieceToStateIndex[BISHOP] = black_hand_bishop;
    PieceToStateIndex[ROOK] = black_hand_rook;
    PieceToStateIndex[BLACK_PAWN] = black_pawn;
    PieceToStateIndex[BLACK_LANCE] = black_lance;
    PieceToStateIndex[BLACK_KNIGHT] = black_knight;
    PieceToStateIndex[BLACK_SILVER] = black_silver;
    PieceToStateIndex[BLACK_GOLD] = black_gold;
    PieceToStateIndex[BLACK_BISHOP] = black_bishop;
    PieceToStateIndex[BLACK_ROOK] = black_rook;
    PieceToStateIndex[BLACK_PAWN_PROMOTE] = black_gold;
    PieceToStateIndex[BLACK_LANCE_PROMOTE] = black_gold;
    PieceToStateIndex[BLACK_KNIGHT_PROMOTE] = black_gold;
    PieceToStateIndex[BLACK_SILVER_PROMOTE] = black_gold;
    PieceToStateIndex[BLACK_BISHOP_PROMOTE] = black_horse;
    PieceToStateIndex[BLACK_ROOK_PROMOTE] = black_dragon;
    PieceToStateIndex[WHITE_PAWN] = white_pawn;
    PieceToStateIndex[WHITE_LANCE] = white_lance;
    PieceToStateIndex[WHITE_KNIGHT] = white_knight;
    PieceToStateIndex[WHITE_SILVER] = white_silver;
    PieceToStateIndex[WHITE_GOLD] = white_gold;
    PieceToStateIndex[WHITE_BISHOP] = white_bishop;
    PieceToStateIndex[WHITE_ROOK] = white_rook;
    PieceToStateIndex[WHITE_PAWN_PROMOTE] = white_gold;
    PieceToStateIndex[WHITE_LANCE_PROMOTE] = white_gold;
    PieceToStateIndex[WHITE_KNIGHT_PROMOTE] = white_gold;
    PieceToStateIndex[WHITE_SILVER_PROMOTE] = white_gold;
    PieceToStateIndex[WHITE_BISHOP_PROMOTE] = white_horse;
    PieceToStateIndex[WHITE_ROOK_PROMOTE] = white_dragon;
}

PieceState PieceStateIndex[] = {
    black_hand_pawn,
    white_hand_pawn,
    black_hand_lance,
    white_hand_lance,
    black_hand_knight,
    white_hand_knight,
    black_hand_silver,
    white_hand_silver,
    black_hand_gold,
    white_hand_gold,
    black_hand_bishop,
    white_hand_bishop,
    black_hand_rook,
    white_hand_rook,
    black_pawn,
    white_pawn,
    black_lance,
    white_lance,
    black_knight,
    white_knight,
    black_silver,
    white_silver,
    black_gold,
    white_gold,
    black_bishop,
    white_bishop,
    black_rook,
    white_rook,
    black_horse,
    white_horse,
    black_dragon,
    white_dragon,
    square_end,
    PieceStateNum,
};

void initInvPieceState() {
    for (int i = 0; i < 18; ++i) {
        invPieceStateIndex[black_hand_pawn + i] = PieceState(white_hand_pawn + i);
        invPieceStateIndex[white_hand_pawn + i] = PieceState(black_hand_pawn + i);
    }
    
    for (int i = 0; i < 4; ++i) {
        invPieceStateIndex[black_hand_lance + i] = PieceState(white_hand_lance + i);
        invPieceStateIndex[black_hand_knight + i] = PieceState(white_hand_knight + i);
        invPieceStateIndex[black_hand_silver + i] = PieceState(white_hand_silver + i);
        invPieceStateIndex[black_hand_gold + i] = PieceState(white_hand_gold + i);

        invPieceStateIndex[white_hand_lance + i] = PieceState(black_hand_lance + i);
        invPieceStateIndex[white_hand_knight + i] = PieceState(black_hand_knight + i);
        invPieceStateIndex[white_hand_silver + i] = PieceState(black_hand_silver + i);
        invPieceStateIndex[white_hand_gold + i] = PieceState(black_hand_gold + i);
    }

    for (int i = 0; i < 2; ++i) {
        invPieceStateIndex[black_hand_bishop + i] = PieceState(white_hand_bishop + i);
        invPieceStateIndex[black_hand_rook + i] = PieceState(white_hand_rook + i);

        invPieceStateIndex[white_hand_bishop + i] = PieceState(black_hand_bishop + i);
        invPieceStateIndex[white_hand_rook + i] = PieceState(black_hand_rook + i);
    }

    for (int i = 0; i < 81; ++i) {
        invPieceStateIndex[black_pawn   + i] = PieceState(white_pawn   + 80 - i);
        invPieceStateIndex[black_lance  + i] = PieceState(white_lance  + 80 - i);
        invPieceStateIndex[black_knight + i] = PieceState(white_knight + 80 - i);
        invPieceStateIndex[black_silver + i] = PieceState(white_silver + 80 - i);
        invPieceStateIndex[black_gold   + i] = PieceState(white_gold   + 80 - i);
        invPieceStateIndex[black_bishop + i] = PieceState(white_bishop + 80 - i);
        invPieceStateIndex[black_rook   + i] = PieceState(white_rook   + 80 - i);
        invPieceStateIndex[black_horse  + i] = PieceState(white_horse  + 80 - i);
        invPieceStateIndex[black_dragon + i] = PieceState(white_dragon + 80 - i);

        invPieceStateIndex[white_pawn   + i] = PieceState(black_pawn   + 80 - i);
        invPieceStateIndex[white_lance  + i] = PieceState(black_lance  + 80 - i);
        invPieceStateIndex[white_knight + i] = PieceState(black_knight + 80 - i);
        invPieceStateIndex[white_silver + i] = PieceState(black_silver + 80 - i);
        invPieceStateIndex[white_gold   + i] = PieceState(black_gold   + 80 - i);
        invPieceStateIndex[white_bishop + i] = PieceState(black_bishop + 80 - i);
        invPieceStateIndex[white_rook   + i] = PieceState(black_rook   + 80 - i);
        invPieceStateIndex[white_horse  + i] = PieceState(black_horse  + 80 - i);
        invPieceStateIndex[white_dragon + i] = PieceState(black_dragon + 80 - i);
    }
}

PieceState mirrorPieceState(PieceState ps) {
    if (ps < hand_end) { //持ち駒はそのまま
        return ps;
    }

    int32_t index = -1;
    for (int i = 0; i < 34 - 1; i++) {
        if (PieceStateIndex[i] <= ps && ps < PieceStateIndex[i + 1]) {
            index = PieceStateIndex[i];
            break;
        }
    }
    assert(index != -1);
    int32_t sq_num = ps - index;
    Square mirror_sq = FRToSquare[File9 - sq_num / 9][sq_num % 9 + 1];
    return PieceState(index + SquareToNum[mirror_sq]);
}

std::ostream & operator<<(std::ostream & os, const PieceState ps) {
    static std::unordered_map<int, std::string> dictionary = {
        { black_hand_pawn, "先手持ち歩" },
        { black_hand_lance, "先手持ち香" },
        { black_hand_knight, "先手持ち桂" },
        { black_hand_silver, "先手持ち銀" },
        { black_hand_gold, "先手持ち金" },
        { black_hand_bishop, "先手持ち角" },
        { black_hand_rook, "先手持ち飛" },
        { white_hand_pawn, "後手持ち歩" },
        { white_hand_lance, "後手持ち香" },
        { white_hand_knight, "後手持ち桂" },
        { white_hand_silver, "後手持ち銀" },
        { white_hand_gold, "後手持ち金" },
        { white_hand_bishop, "後手持ち角" },
        { white_hand_rook, "後手持ち飛" },
        { black_pawn, "先手歩" },
        { black_lance, "先手香" },
        { black_knight, "先手桂" },
        { black_silver, "先手銀" },
        { black_gold, "先手金" },
        { black_bishop, "先手角" },
        { black_rook, "先手飛" },
        { black_horse, "先手馬" },
        { black_dragon, "先手竜" },
        { white_pawn, "後手歩" },
        { white_lance, "後手香" },
        { white_knight, "後手桂" },
        { white_silver, "後手銀" },
        { white_gold, "後手金" },
        { white_bishop, "後手角" },
        { white_rook, "後手飛" },
        { white_horse, "後手馬" },
        { white_dragon, "後手竜" },
    };

    int ps_index = -1;
    for (int i = 0; i < 34 - 1; i++) {
        if (PieceStateIndex[i] <= ps && ps < PieceStateIndex[i + 1]) {
            ps_index = PieceStateIndex[i];
            break;
        }
    }
    assert(ps_index != -1);

    //ps < hand_endなら枚数 - 1を示すし、そうでないならマスの位置を示す
    int offset = ps - ps_index;

    if (ps < hand_end) {
        os << dictionary[PieceState(ps_index)] << ":" << offset + 1;
    } else {
        os << offset / 9 + 1 << offset % 9 + 1 << dictionary[PieceState(ps_index)];
    }
    return os;
}

PieceState pieceState(const Piece p, int square_or_num, Color c) {
    if (p <= KING) { //持ち駒
        if (p == KING) {
            std::cout << "玉を取る手が発生" << std::endl;
            assert(false);
        }
        assert(square_or_num >= 1);
        //持ち駒
        --square_or_num;

        if (c == BLACK) {
            return PieceState(PieceToStateIndex[p] + square_or_num);
        }  else {
            //後手の分だけずらす
            int ps = (PieceToStateIndex[p] + square_or_num);
            switch (p) {
            case PAWN:
                ps += 18; break;
            case LANCE:
            case KNIGHT:
            case SILVER:
            case GOLD:
                ps += 4; break;
            case BISHOP:
            case ROOK:
                ps += 2; break;
            default:
                assert(false);
            }
            return PieceState(ps);
        }
    } else { //盤上の駒
        return PieceState(PieceToStateIndex[p] + SquareToNum[square_or_num]);
    }
}