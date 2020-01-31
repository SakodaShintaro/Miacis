#ifndef MIACIS_MOVE_HPP
#define MIACIS_MOVE_HPP

#include"square.hpp"
#include"piece.hpp"
#include"../types.hpp"
#include<unordered_map>
#include<iostream>

//行動の次元数
//1ch目は普通の行動,2ch目はパス専用
constexpr int64_t POLICY_CHANNEL_NUM = 2;
constexpr int64_t POLICY_DIM = SQUARE_NUM * POLICY_CHANNEL_NUM;

class Move {
public:
    static constexpr int32_t TURN_BIT = 10;

    //コンストラクタ
    Move() = default;
    Move(int32_t x) : move(x) {}
    Move(Square to) : move(to) {}
    Move(Square to, Color c) : move(c == BLACK ? to : to | (1 << TURN_BIT)) {}

    //表示
    void print() const {
        std::cout << to() << std::endl;
    }

    //要素を取り出す関数ら
    inline Square to() const { return Square(move & ~(1 << TURN_BIT)); }
    inline Color color() const { return (move & (1 << TURN_BIT) ? WHITE : BLACK); }

    //演算子オーバーロード
    bool operator==(const Move &rhs) const { return (move == rhs.move); }
    bool operator!=(const Move &rhs) const { return !(*this == rhs); }

    //ラベル系
    //行動から教師ラベルへと変換する関数
    int32_t toLabel() const {
        return (move == 0 ? SQUARE_NUM : SquareToNum[to()]);
    }
    //ラベルを左右反転させる関数。左右反転のデータ拡張に対応するために必要
    static uint32_t augmentedLabel(uint32_t label, int64_t augmentation) {
        assert(augmentation == 0);
        return label;
    }

    //探索時にSearcherクラスから気軽にアクセスできるようpublicにおいてるけど
    int32_t move;
};

//比較用
const Move NULL_MOVE(0);

//ストリームに対して出力するオーバーロード
inline std::ostream& operator<<(std::ostream& os, Move m) {
    if (m == NULL_MOVE) {
        os << "PA";
    } else {
        os << fileToString[SquareToFile[m.to()]] << SquareToRank[m.to()];
    }
    return os;
}

inline Move stringToMove(std::string input) {
    if (input == "PA" || input == "PS") {
        return NULL_MOVE;
    }
    Square to = FRToSquare[File8 - (input[0] - 'A')][input[1] - '0'];
    return Move(to);
}

#endif //MIACIS_MOVE_HPP