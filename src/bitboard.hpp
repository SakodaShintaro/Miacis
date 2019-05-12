#ifndef BITBOARD_HPP
#define BITBOARD_HPP

#include"square.hpp"
#include"common.hpp"
#include<bitset>

class Bitboard {
public:
	//引数なしコンストラクタは空でいいかな
	Bitboard() = default;

	//値を直接引数に持つコンストラクタ
    Bitboard(uint64_t b0, uint64_t b1) : board_{ b0, b1 } {}

	//Squareを指定してそこだけを立てるコンストラクタ
    explicit Bitboard(Square sq);

    explicit operator bool() const {
        return !(board_[0] == 0 && board_[1] == 0);
    }

    uint64_t merge() const {
        return board_[0] | board_[1];
    }

    bool crossOver() const {
        return bool(board_[0] & board_[1]);
    }

    Square pop() {
        Square sq;
        if (board_[0] != 0) {
            sq = SquareList[pop_lsb(board_[0])];
        } else {
            sq = SquareList[pop_lsb(board_[1]) + 63];
        }
        assert(isOnBoard(sq));
        return sq;
    }

    auto pop_count() const {
        return POP_CNT64(board_[0]) + POP_CNT64(board_[1]);
    }

    template<class Function>
    void forEach(Function f) const {
        Bitboard copy = *this;
        while (copy) {
            Square sq = copy.pop();
            f(sq);
        }
    }

	//演算子類
	Bitboard operator ~() const {
        return Bitboard(~board_[0], ~board_[1]);
    }
	Bitboard operator |(const Bitboard& bb) const {
		return Bitboard(board_[0] | bb.board_[0], board_[1] | bb.board_[1]);
	}
	Bitboard operator |(const Square sq) {
		return *this | Bitboard(sq);
	}
    Bitboard operator &(const Bitboard& bb) const {
        return Bitboard(board_[0] & bb.board_[0], board_[1] & bb.board_[1]);
    }
    Bitboard& operator|=(const Bitboard& rhs) {
        board_[0] |= rhs.board_[0];
        board_[1] |= rhs.board_[1];
        return *this;
    }
    Bitboard& operator&=(const Bitboard& rhs) {
        board_[0] &= rhs.board_[0];
        board_[1] &= rhs.board_[1];
        return *this;
    }
    Bitboard& operator <<= (const int shift) {
        board_[0] <<= shift;
        board_[1] <<= shift;
        return *this;
    }
    Bitboard operator << (const int shift) {
        return Bitboard(*this) <<= shift;
    }

    static void init();

    static int part(const Square sq) {
        return (sq > SQ79 ? 1 : 0);
    }

	//bitレイアウトはboard_[0]に1筋から7筋まで、board_[1]に残りの8,9筋を入れたい
	//64bit目は不使用？

    /*
	union {
		uint64_t board_[2];
		__m128i m_;
	};
    */
    uint64_t board_[2];
};

extern Bitboard BOARD_BB;
extern Bitboard SQUARE_BB[SquareNum];
extern Bitboard FILE_BB[FileNum];
extern Bitboard RANK_BB[RankNum];
extern Bitboard PROMOTION_ZONE_BB[ColorNum];
extern Bitboard FRONT_BB[ColorNum][RankNum];
extern Bitboard BETWEEN_BB[SquareNum][SquareNum];

extern Bitboard PAWN_CONTROL_BB[ColorNum][SquareNum];
extern Bitboard KNIGHT_CONTROL_BB[ColorNum][SquareNum];
extern Bitboard SILVER_CONTROL_BB[ColorNum][SquareNum];
extern Bitboard GOLD_CONTROL_BB[ColorNum][SquareNum];

extern Bitboard BishopEffect[2][SquareNum][128];
extern Bitboard BishopEffectMask[2][SquareNum];

extern uint64_t RookFileEffect[RankNum][128];
extern Bitboard RookRankEffect[FileNum][128];

extern Bitboard KING_CONTROL_BB[SquareNum];

extern int Slide[];

std::ostream& operator << (std::ostream& os, const Bitboard& rhs);

inline Bitboard blackPawnControl(const Square sq, const Bitboard &occ) {
    return PAWN_CONTROL_BB[BLACK][sq];
}

inline Bitboard whitePawnControl(const Square sq, const Bitboard &occ) {
    return PAWN_CONTROL_BB[WHITE][sq];
}

inline Bitboard blackNightControl(const Square sq, const Bitboard& occ) {
    return KNIGHT_CONTROL_BB[BLACK][sq];
}

inline Bitboard whiteNightControl(const Square sq, const Bitboard& occ) {
    return KNIGHT_CONTROL_BB[WHITE][sq];
}

inline Bitboard blackSilverControl(const Square sq, const Bitboard& occ) {
    return SILVER_CONTROL_BB[BLACK][sq];
}

inline Bitboard whiteSilverControl(const Square sq, const Bitboard& occ) {
    return SILVER_CONTROL_BB[WHITE][sq];
}

inline Bitboard blackGoldControl(const Square sq, const Bitboard& occ) {
    return GOLD_CONTROL_BB[BLACK][sq];
}

inline Bitboard whiteGoldControl(const Square sq, const Bitboard& occ) {
    return GOLD_CONTROL_BB[WHITE][sq];
}

inline uint64_t occupiedToIndex(const Bitboard& occupied, const Bitboard& mask) {
    return PEXT64(occupied.merge(), mask.merge());
}

inline Bitboard bishopEffect0(const Square sq, const Bitboard& occupied) {
    const Bitboard block0(occupied & BishopEffectMask[0][sq]);
    return BishopEffect[0][sq][occupiedToIndex(block0, BishopEffectMask[0][sq])];
}

inline Bitboard bishopEffect1(const Square sq, const Bitboard& occupied) {
    const Bitboard block1(occupied & BishopEffectMask[1][sq]);
    return BishopEffect[1][sq][occupiedToIndex(block1, BishopEffectMask[1][sq])];
}

inline Bitboard bishopControl(const Square sq, const Bitboard& occupied) {
    return bishopEffect0(sq, occupied) | bishopEffect1(sq, occupied);
}

inline Bitboard rookFileControl(const Square sq, const Bitboard& occupied) {
    const auto index = (occupied.board_[Bitboard::part(sq)] >> Slide[sq]) & 0x7f;
    const File f = SquareToFile[sq];
    return (f <= File7) ?
        Bitboard(RookFileEffect[SquareToRank[sq]][index] << (9 * (f - File1)), 0) :
        Bitboard(0, RookFileEffect[SquareToRank[sq]][index] << (9 * (f - File8)));
}

inline Bitboard rookRankControl(const Square sq, const Bitboard& occupied) {
    int r = SquareToRank[sq];
    uint64_t u = (occupied.board_[1] << 6 * 9) + (occupied.board_[0] >> 9);
    uint64_t index = PEXT64(u, (uint64_t)(0b1000000001000000001000000001000000001000000001000000001 << (r - Rank1)));
    return RookRankEffect[SquareToFile[sq]][index] << (r - Rank1);
}

inline Bitboard rookControl(const Square sq, const Bitboard& occupied) {
    return rookFileControl(sq, occupied) | rookRankControl(sq, occupied);
}

inline Bitboard lanceControl(const Color color, const Square sq, const Bitboard& occupied) {
    return rookControl(sq, occupied) & FRONT_BB[color][SquareToRank[sq]];
}

inline Bitboard blackLanceControl(const Square sq, const Bitboard& occ) {
    return lanceControl(BLACK, sq, occ);
}

inline Bitboard whiteLanceControl(const Square sq, const Bitboard& occ) {
    return lanceControl(WHITE, sq, occ);
}

inline Bitboard kingControl(const Square sq, const Bitboard& occ) {
    return KING_CONTROL_BB[sq];
}

inline Bitboard horseControl(const Square sq, const Bitboard& occupied) {
    return bishopControl(sq, occupied) | kingControl(sq, occupied);
}

inline Bitboard dragonControl(const Square sq, const Bitboard& occupied) {
    return rookControl(sq, occupied) | kingControl(sq, occupied);
}

static Bitboard(*controlFunc[])(const Square sq, const Bitboard& occupied) = {
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, //0~9
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, //10~19
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, //20~29
    nullptr, nullptr, nullptr, //30~32
    blackPawnControl,  //33
    blackLanceControl, //34
    blackNightControl, //35
    blackSilverControl,//36
    blackGoldControl,  //37
    bishopControl,     //38
    rookControl,       //39
    kingControl,       //40
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, //40~48
    blackGoldControl,  //49 と
    blackGoldControl,  //50 成香
    blackGoldControl,  //51 成桂
    blackGoldControl,  //52 成銀
    nullptr,           //53 成金などない
    horseControl,      //54 馬
    dragonControl,     //55 竜
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, //56~64
    whitePawnControl,  //65
    whiteLanceControl, //66
    whiteNightControl, //67
    whiteSilverControl,//68
    whiteGoldControl,  //69
    bishopControl,     //70
    rookControl,       //71
    kingControl,       //72
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, //73~80
    whiteGoldControl,  //81 と
    whiteGoldControl,  //82 成香
    whiteGoldControl,  //83 成桂
    whiteGoldControl,  //84 成銀
    nullptr,           //85 成金などない
    horseControl,      //86 馬
    dragonControl,     //87 竜
};

inline Bitboard controlBB(const Square sq, const Piece p, const Bitboard& occupied) {
    return controlFunc[p](sq, occupied);
}

//手番側から見て一番奥の段を返す(駒打ちの時に利用)
inline Bitboard farRank1FromColor(const Color c) {
    return (c == BLACK ? RANK_BB[Rank1] : RANK_BB[Rank9]);
}

//手番側から見て奥から2つ目の段を返す(駒打ちの時に利用)
inline Bitboard farRank2FromColor(const Color c) {
    return (c == BLACK ? RANK_BB[Rank2] : RANK_BB[Rank8]);
}

#endif // !BITBOARD_HPP