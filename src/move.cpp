#include"move.hpp"

std::unordered_map<int32_t, int32_t> Move::mp;

uint32_t Move::toLabel() const {
    Color c = pieceToColor(subject());
    Square to_sq = (c == BLACK ? to() : InvSquare[to()]);
    Square from_sq = (c == BLACK ? from() : InvSquare[from()]);

    //移動先のマス
    int32_t to_num = SquareToNum[to_sq];

    //移動元からの方向
    int32_t direction;
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
    }
    if (isPromote()) {
        direction += 10;
    }

    return static_cast<uint32_t>(to_num + SQUARE_NUM * direction);
}

int32_t Move::toID() const {
    assert(mp.count(move & ~MOVE_CAPTURE_MASK) == 1);
    return mp[move & ~MOVE_CAPTURE_MASK];
}