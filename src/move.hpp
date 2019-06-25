﻿#ifndef MOVE_HPP
#define MOVE_HPP

#include"square.hpp"
#include"piece.hpp"
#include"types.hpp"
#include<unordered_map>
#include<iostream>

enum MoveConst {
    //0000 0000 0000 0000 0000 0000 0111 1111 to
    //0000 0000 0000 0000 0011 1111 1000 0000 from
    //0000 0000 0000 0000 0100 0000 0000 0000 drop
    //0000 0000 0000 0000 1000 0000 0000 0000 promote
    //0000 0000 1111 1111 0000 0000 0000 0000 subject
    //1111 1111 0000 0000 0000 0000 0000 0000 capture
    MOVE_TO_SHIFT = 0,
    MOVE_FROM_SHIFT = 7,
    MOVE_DROP_SHIFT = 14,
    MOVE_PROMOTE_SHIFT = 15,
    MOVE_SUBJECT_SHIFT = 16,
    MOVE_CAPTURE_SHIFT = 24,
    MOVE_TO_MASK = 0b1111111,
    MOVE_FROM_MASK = MOVE_TO_MASK << MOVE_FROM_SHIFT,
    MOVE_DROP_MASK = 1 << MOVE_DROP_SHIFT,
    MOVE_PROMOTE_MASK = 1 << MOVE_PROMOTE_SHIFT,
    MOVE_SUBJECT_MASK = 0xff << MOVE_SUBJECT_SHIFT,
    MOVE_CAPTURE_MASK = 0xff << MOVE_CAPTURE_SHIFT,
};

class Move {
public:
    //コンストラクタ
    Move() = default;

    explicit Move(int x) : move(x), score(Score(0)) {}

    Move(Square to, Square from) : move(from << MOVE_FROM_SHIFT
                                      | to << MOVE_TO_SHIFT) {}

    Move(Square to, Square from, bool isDrop) : move(isDrop << MOVE_DROP_SHIFT
                                                   | from << MOVE_FROM_SHIFT
                                                   | to << MOVE_TO_SHIFT) {}

    Move(Square to, Square from, bool isDrop, bool isPromote) : move(isPromote << MOVE_PROMOTE_SHIFT
                                                                   | isDrop << MOVE_DROP_SHIFT
                                                                   | from << MOVE_FROM_SHIFT
                                                                   | to << MOVE_TO_SHIFT) {}

    Move(Square to, Square from, bool isDrop, bool isPromote, Piece subject) : move(subject << MOVE_SUBJECT_SHIFT
                                                                                  | isPromote << MOVE_PROMOTE_SHIFT
                                                                                  | isDrop << MOVE_DROP_SHIFT
                                                                                  | from << MOVE_FROM_SHIFT
                                                                                  | to << MOVE_TO_SHIFT) {}

    Move(Square to, Square from, bool isDrop, bool isPromote, Piece subject, Piece capture) : move(
              capture << MOVE_CAPTURE_SHIFT
            | subject << MOVE_SUBJECT_SHIFT
            | isPromote << MOVE_PROMOTE_SHIFT
            | isDrop << MOVE_DROP_SHIFT
            | from << MOVE_FROM_SHIFT
            | to << MOVE_TO_SHIFT) {}

    void print() const {
        std::cout << SquareToFile[to()] << SquareToRank[to()] << PieceToStr[subject()];
        if (isPromote()) {
            std::cout << "成";
        }
        if (isDrop()) {
            std::cout << "打";
        } else {
            std::cout << "(" << SquareToFile[from()] << SquareToRank[from()] << ") ";
        }
        if (capture() != EMPTY) {
            std::cout << "capture:" << PieceToStr[capture()];
        }
    }

    //見やすい表示
    void printWithNewLine() const {
        print();
        std::cout << std::endl;
    }

    //要素を取り出す関数ら
    inline Square to() const { return static_cast<Square>(move & MOVE_TO_MASK); }
    inline Square from() const { return static_cast<Square>((move & MOVE_FROM_MASK) >> MOVE_FROM_SHIFT); }
    inline bool isDrop() const { return (move & MOVE_DROP_MASK) != 0; }
    inline bool isPromote() const { return (move & MOVE_PROMOTE_MASK) != 0; }
    inline Piece subject() const { return static_cast<Piece>((move & MOVE_SUBJECT_MASK) >> MOVE_SUBJECT_SHIFT); }
    inline Piece capture() const { return static_cast<Piece>((move & MOVE_CAPTURE_MASK) >> MOVE_CAPTURE_SHIFT); }

    //演算子オーバーロード
    bool operator==(const Move& rhs) const { return (move == rhs.move); }
    bool operator!=(const Move& rhs) const { return !(*this == rhs); }
    bool operator<(const Move& rhs) const { return (score < rhs.score); }
    bool operator>(const Move& rhs) const { return (score > rhs.score); }

    uint32_t toLabel() const;

    //探索時にSearcherクラスから気軽にアクセスできるようpublicにおいてるけど
    int32_t move;
    Score score;
};

//駒を打つ手
inline Move dropMove(Square to, Piece p) { return Move(to, WALL00, true, false, p, EMPTY); }

//駒を動かす手を引数として、それの成った動きを返す
inline Move promotiveMove(Move non_promotive_move) {
    return Move(non_promotive_move.to(), non_promotive_move.from(), false, true, non_promotive_move.subject(),
                non_promotive_move.capture());
}

//比較用
const Move NULL_MOVE(0);

//sfen形式で出力するオーバーロード
inline std::ostream& operator<<(std::ostream& os, Move m) {
    if (m == NULL_MOVE) {
        os << "resign";
        return os;
    }
    if (m.isDrop()) {
        os << PieceToSfenStr[kind(m.subject())][0] << '*' << static_cast<int>(SquareToFile[m.to()])
           << static_cast<char>(SquareToRank[m.to()] + 'a' - 1);
    } else {
        os << static_cast<int>(SquareToFile[m.from()])
           << static_cast<char>(SquareToRank[m.from()] + 'a' - 1)
           << static_cast<int>(SquareToFile[m.to()])
           << static_cast<char>(SquareToRank[m.to()] + 'a' - 1);
        if (m.isPromote()) {
            os << '+';
        }
    }
    return os;
}

//これコンストラクタとかで書いた方がいい気がするけどうまく書き直せなかった
//まぁ動けばいいのかなぁ
static Move stringToMove(std::string input) {
    static std::unordered_map<char, Piece> charToPiece = {
            {'P', PAWN},
            {'L', LANCE},
            {'N', KNIGHT},
            {'S', SILVER},
            {'G', GOLD},
            {'B', BISHOP},
            {'R', ROOK},
    };
    if ('A' <= input[0] && input[0] <= 'Z') { //持ち駒を打つ手
        Square to = FRToSquare[input[2] - '0'][input[3] - 'a' + 1];
        return dropMove(to, charToPiece[input[0]]);
    } else { //盤上の駒を動かす手
        Square from = FRToSquare[input[0] - '0'][input[1] - 'a' + 1];
        Square to = FRToSquare[input[2] - '0'][input[3] - 'a' + 1];
        bool promote = (input.size() == 5 && input[4] == '+');
        return Move(to, from, false, promote, EMPTY, EMPTY);
    }
}

#endif