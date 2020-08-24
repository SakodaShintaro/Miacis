#ifndef MIACIS_GO_MOVE_HPP
#define MIACIS_GO_MOVE_HPP

#include "../types.hpp"
#include "piece.hpp"
#include "square.hpp"
#include <iostream>
#include <sstream>

namespace Go {

//行動の次元数
//1ch目は普通の行動,2ch目はパス専用
constexpr int64_t POLICY_CHANNEL_NUM = 2;
constexpr int64_t POLICY_DIM = SQUARE_NUM * POLICY_CHANNEL_NUM;

class Move {
public:
    static constexpr int32_t TURN_BIT = 10;

    //コンストラクタ
    Move() = default;
    explicit Move(Square to) : move(to) {}
    Move(Square to, Color c) : move(c == BLACK ? to : to | (1 << TURN_BIT)) {}

    //見やすい日本語での表示
    std::string toPrettyStr() const { return (move == SQUARE_NUM ? "PASS" : squareToString(to())); }

    //要素を取り出す関数ら
    inline Square to() const { return Square(move & ~(1 << TURN_BIT)); }
    inline Color color() const { return (move & (1 << TURN_BIT) ? WHITE : BLACK); }

    //演算子オーバーロード
    bool operator==(const Move& rhs) const { return (move == rhs.move); }
    bool operator!=(const Move& rhs) const { return !(*this == rhs); }

    //ラベル系
    //行動から教師ラベルへと変換する関数
    int32_t toLabel() const { return to(); }
    //ラベルをデータ拡張に対応させる関数
    static uint32_t augmentLabel(uint32_t label, int64_t augmentation) {
        if (label == SQUARE_NUM) {
            //SQUARE_NUMはパスを示している。それはどのように反転しても同じ
            return SQUARE_NUM;
        }
        return augmentLabelMirror(augmentLabelRotate(label, augmentation % 4), augmentation / 4);
    }

    //探索時にSearcherクラスから気軽にアクセスできるようpublicにおいてるけど
    int32_t move;

private:
    static uint32_t augmentLabelRotate(uint32_t label, int64_t augmentation) {
        if (augmentation == 0) {
            //なにもしない
            return label;
        } else if (augmentation == 1) {
            //時計回りに90度回転
            int64_t rank = label % BOARD_WIDTH;
            int64_t file = label / BOARD_WIDTH;
            return rank * BOARD_WIDTH + BOARD_WIDTH - 1 - file;
        } else if (augmentation == 2) {
            //時計回りに180度回転
            return SQUARE_NUM - 1 - label;
        } else if (augmentation == 3) {
            //時計回りに270度回転
            int64_t rank = label % BOARD_WIDTH;
            int64_t file = label / BOARD_WIDTH;
            return (BOARD_WIDTH - 1 - rank) * BOARD_WIDTH + file;
        } else {
            std::cout << "in augmentLabelRotate, augmentation = " << augmentation << std::endl;
            exit(1);
        }
    }
    static uint32_t augmentLabelMirror(uint32_t label, int64_t augmentation) {
        if (augmentation == 0) {
            //なにもしない
            return label;
        } else if (augmentation == 1) {
            //左右反転
            int64_t rank = label % BOARD_WIDTH;
            int64_t file = label / BOARD_WIDTH;
            return (BOARD_WIDTH - 1 - file) * BOARD_WIDTH + rank;
        } else {
            std::cout << "in augmentLabelMirror, augmentation = " << augmentation << std::endl;
            exit(1);
        }
    }
};

//比較用
const Move NULL_MOVE(SQUARE_NUM);

//ストリームに対して出力するオーバーロード
inline std::ostream& operator<<(std::ostream& os, Move m) {
    os << m.toPrettyStr();
    return os;
}

inline Move stringToMove(std::string input) {
    if (input == "PA" || input == "PS" || input == "pass" || input == "pa" || input == "PASS") {
        return NULL_MOVE;
    }

    if (input.size() != 2) {
        input = input.substr(0, 2);
    }

    char x = input[0];
    char y = input[1];

    if (!('1' <= y && y < '1' + BOARD_WIDTH)) {
        return NULL_MOVE;
    }

    //'I'は遣わず一つずつズレるのでここでズラす
    if (x > 'I') {
        x--;
    }

    if ('A' <= x && x < 'A' + BOARD_WIDTH) {
        Square to = xy2square(x - 'A', y - '1');
        return Move(to);
    } else if ('a' <= x && x < 'a' + BOARD_WIDTH) {
        Square to = xy2square(x - 'A', y - '1');
        return Move(to);
    } else {
        return NULL_MOVE;
    }
}

} // namespace Go

#endif //MIACIS_GO_MOVE_HPP