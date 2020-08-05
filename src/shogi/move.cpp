#include"move.hpp"
#include"position.hpp"

uint32_t Move::toLabel() const {
    Color c = pieceToColor(subject());
    Square to_sq = (c == BLACK ? to() : InvSquare[to()]);
    Square from_sq = (c == BLACK ? from() : InvSquare[from()]);

    //移動先のマス
    int32_t to_num = SquareToNum[to_sq];

    //移動元からの方向
    int32_t direction = {};
    File to_file = SquareToFile[to_sq];
    Rank to_rank = SquareToRank[to_sq];
    File from_file = SquareToFile[from_sq];
    Rank from_rank = SquareToRank[from_sq];

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

    return static_cast<uint32_t>(to_num + SQUARE_NUM * direction);
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

void Move::print() const {
    std::cout << toPrettyStr() << std::endl;
}

std::string Move::toPrettyStr() const {
    if (move_ == MOVE_DECLARE) {
        return "入玉宣言";
    }
    std::stringstream str;
    str << SquareToFile[to()] << SquareToRank[to()] << PieceToStr[subject()];
    if (isPromote()) {
        str << "成";
    }
    if (isDrop()) {
        str << "打";
    } else {
        str << "(" << SquareToFile[from()] << SquareToRank[from()] << ") ";
    }
    if (capture() != EMPTY) {
        str << "capture:" << PieceToStr[capture()];
    }
    return str.str();
}