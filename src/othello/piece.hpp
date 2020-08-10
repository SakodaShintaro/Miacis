#ifndef MIACIS_PIECE_HPP
#define MIACIS_PIECE_HPP

#include "../array_map.hpp"
#include "../types.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

enum Piece : int32_t { BLACK_PIECE, WHITE_PIECE, EMPTY, WALL, PieceNum };

extern const ArrayMap<std::string, PieceNum> PieceToStr;
extern const ArrayMap<std::string, PieceNum> PieceToSfenStr;

inline Piece operator++(Piece& p, int) { return p = static_cast<Piece>(p + 1); }
inline Piece operator|(Piece lhs, Piece rhs) { return Piece(int(lhs) | int(rhs)); }
inline int operator<<(Piece p, int shift) { return static_cast<int>(p) << shift; }

//空のマスに対するチャンネルを用意して入力する必要はあるんだろうか？
constexpr int64_t INPUT_CHANNEL_NUM = 2;

inline Piece oppositeColor(Piece p) { return (p == BLACK_PIECE ? WHITE_PIECE : BLACK_PIECE); }

std::ostream& operator<<(std::ostream&, Piece piece);
#endif //MIACIS_PIECE_HPP