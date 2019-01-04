#pragma once
#ifndef EVAL_ELEMENTS_HPP
#define EVAL_ELEMENTS_HPP

#include"piece.hpp"
#include"piece_state.hpp"
#include"square.hpp"
#include"eval_params.hpp"
#include<algorithm>
#include<vector>

constexpr int PIECE_STATE_LIST_SIZE = 38;

//評価値の計算に用いる特徴量をまとめたもの
#ifdef USE_NN
using Features = std::vector<float>;

#else
struct alignas(32) Features {
    //先手から見たもの[0] 後手からみたもの[1] を両方持つ
    //アラインメントが合うようにPIECE_STATE_LIST_SIZE + 2だけ確保する
    PieceState piece_state_list[ColorNum][PIECE_STATE_LIST_SIZE + 2];
    Square king_sq[ColorNum];
    Color color;

    inline bool operator==(const Features& rhs) {
        return (king_sq[BLACK] == rhs.king_sq[BLACK]
            && king_sq[WHITE] == rhs.king_sq[WHITE]
            && piece_state_list == rhs.piece_state_list);
    }
    inline bool operator!=(const Features& rhs) {
        return !(*this == rhs);
    }
};

inline bool isEqual(Features f1, Features f2) {
    //順番をそろえるためにソートしてpiece_state_listの中身を比較する比較関数
    for (int32_t c = 0; c < ColorNum; c++) {
        std::sort(&f1.piece_state_list[c][0], &f1.piece_state_list[c][0] + PIECE_STATE_LIST_SIZE);
        std::sort(&f2.piece_state_list[c][0], &f2.piece_state_list[c][0] + PIECE_STATE_LIST_SIZE);
        for (int32_t i = 0; i < PIECE_STATE_LIST_SIZE; i++) {
            if (f1.piece_state_list[c][i] != f2.piece_state_list[c][i]) {
                return false;
            }
        }
    }
    return (f1.king_sq[BLACK] == f2.king_sq[BLACK] && f1.king_sq[WHITE] == f2.king_sq[WHITE] && f1.color == f2.color);
}

inline std::ostream& operator<<(std::ostream& os, Features f) {
    os << "先手玉:" << f.king_sq[BLACK] << std::endl;
    os << "後手玉:" << f.king_sq[WHITE] << std::endl;
    for (auto e : f.piece_state_list) {
        os << e << std::endl;
    }
    return os;
}

#endif

#endif // !EVAL_ELEMENT_HPP
