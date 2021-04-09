#include "move.hpp"
#include "position.hpp"

namespace Shogi {

uint32_t Move::toLabel() const {
    Color c = pieceToColor(subject());

    Square to_sq = (c == BLACK ? to() : InvSquare[to()]);
    File to_file = SquareToFile[to_sq];
    Rank to_rank = SquareToRank[to_sq];
    Square from_sq = (c == BLACK ? from() : InvSquare[from()]);
    File from_file = SquareToFile[from_sq];
    Rank from_rank = SquareToRank[from_sq];

    int32_t direction = {};
    if (from() == WALL00) { //打つ手
        direction = 20 + kind(subject()) - PAWN;
    } else if (to_file == from_file - 1 && to_rank == from_rank + 2) { //桂馬
        direction = 4;
    } else if (to_file == from_file + 1 && to_rank == from_rank + 2) { //桂馬
        direction = 6;
    } else if (to_file == from_file && to_rank > from_rank) { //上
        direction = 0;
    } else if (to_file > from_file && to_rank > from_rank) { //右上
        direction = 1;
    } else if (to_file > from_file && to_rank == from_rank) { //右
        direction = 2;
    } else if (to_file > from_file && to_rank < from_rank) { //右下
        direction = 3;
    } else if (to_file == from_file && to_rank < from_rank) { //下
        direction = 5;
    } else if (to_file < from_file && to_rank < from_rank) { //左下
        direction = 7;
    } else if (to_file < from_file && to_rank == from_rank) { //左
        direction = 8;
    } else if (to_file < from_file && to_rank > from_rank) { //左上
        direction = 9;
    } else {
        assert(false);
    }
    if (isPromote()) {
        direction += 10;
    }

    return static_cast<uint32_t>(SquareToNum[to_sq] + SQUARE_NUM * direction);
}

uint32_t Move::toDLShogiLabel() const {
    Color c = pieceToColor(subject());

    Square to_sq = (c == BLACK ? to() : InvSquare[to()]);
    File to_file = SquareToFile[to_sq];
    Rank to_rank = SquareToRank[to_sq];
    Square from_sq = (c == BLACK ? from() : InvSquare[from()]);
    File from_file = SquareToFile[from_sq];
    Rank from_rank = SquareToRank[from_sq];

    enum MOVE_DIRECTION {
        UP,
        UP_LEFT,
        UP_RIGHT,
        LEFT,
        RIGHT,
        DOWN,
        DOWN_LEFT,
        DOWN_RIGHT,
        UP2_LEFT,
        UP2_RIGHT,
        MOVE_DIRECTION_NUM
    };

    int32_t direction{};
    if (from() == WALL00) { //打つ手
        direction = 2 * MOVE_DIRECTION_NUM + DLShogiPieceToIndex[kind(subject())] - 1;
    } else if (to_file == from_file - 1 && to_rank == from_rank - 2) { //桂馬右上
        direction = UP2_RIGHT;
    } else if (to_file == from_file + 1 && to_rank == from_rank - 2) { //桂馬左上
        direction = UP2_LEFT;
    } else if (to_file == from_file && to_rank > from_rank) {
        direction = DOWN;
    } else if (to_file > from_file && to_rank > from_rank) {
        direction = DOWN_LEFT;
    } else if (to_file > from_file && to_rank == from_rank) {
        direction = LEFT;
    } else if (to_file > from_file && to_rank < from_rank) {
        direction = UP_LEFT;
    } else if (to_file == from_file && to_rank < from_rank) {
        direction = UP;
    } else if (to_file < from_file && to_rank < from_rank) {
        direction = UP_RIGHT;
    } else if (to_file < from_file && to_rank == from_rank) {
        direction = RIGHT;
    } else if (to_file < from_file && to_rank > from_rank) {
        direction = DOWN_RIGHT;
    } else {
        assert(false);
    }
    if (isPromote()) {
        direction += MOVE_DIRECTION_NUM;
    }

    return static_cast<uint32_t>(SquareToNum[to_sq] + SQUARE_NUM * direction);
}

uint32_t Move::augmentLabel(uint32_t label, int64_t augmentation) {
    if (augmentation == 0) {
        //0のときはそのまま
        return label;
    }
    if (augmentation >= Position::DATA_AUGMENTATION_PATTERN_NUM) {
        std::cout << "augmentation = " << augmentation << std::endl;
        exit(1);
    }

    //augmentation == 1のときは左右反転
    //上のtoLabel関数のようにlabelはto_num + SQUARE_NUM * directionとなっている

    //(1)行き先マスを反転
    //SQUARE_NUMでの剰余を取ればマスの数字が得られる
    int32_t sq_num = label % SQUARE_NUM;

    //筋と段を取得(それぞれ0 ~ 8)
    int32_t file = sq_num / 9;
    int32_t rank = sq_num % 9;

    int32_t mirror_sq_num = SquareToNum[FRToSquare[9 - file][rank + 1]];

    //(2)移動の仕方(方向)を反転
    int32_t direction = label / SQUARE_NUM;

    //20以上のときは持ち駒から打つ手なので反転の必要はない
    if (direction < 20) {
        //tマスに対する移動方向元と数字の対応は
        //901
        //8t2
        //753
        //6 4
        //であり、成る手だと+10されるので以下のようにすれば左右反転
        direction = (10 - direction % 10) % 10 + 10 * (direction >= 10);
    }

    return static_cast<uint32_t>(mirror_sq_num + SQUARE_NUM * direction);
}

std::string Move::toPrettyStr() const {
    if (move_ == MOVE_DECLARE) {
        return "入玉宣言";
    }
    std::stringstream str;
    Color c = pieceToColor(subject());
    str << (c == BLACK ? "▲" : "△") << fileToString[SquareToFile[to()]] << rankToString[SquareToRank[to()]]
        << PieceToStr[kindWithPromotion(subject())];
    if (isPromote()) {
        str << "成";
    }
    if (isDrop()) {
        str << "打";
    } else {
        str << "(" << SquareToFile[from()] << SquareToRank[from()] << ") ";
    }
    //    if (capture() != EMPTY) {
    //        str << "capture:" << PieceToStr[capture()];
    //    }
    return str.str();
}

} // namespace Shogi