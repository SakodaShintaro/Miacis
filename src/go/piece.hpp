#ifndef MIACIS_GO_PIECE_HPP
#define MIACIS_GO_PIECE_HPP

#include "../array_map.hpp"
#include "../types.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

namespace Go {

enum Piece : int32_t { BLACK_PIECE, WHITE_PIECE, EMPTY, WALL, PieceNum };

extern const ArrayMap<std::string, PieceNum> PieceToStr;
extern const ArrayMap<std::string, PieceNum> PieceToSfenStr;

inline Piece operator++(Piece& p, int32_t) { return p = static_cast<Piece>(p + 1); }
inline Piece operator|(Piece lhs, Piece rhs) { return static_cast<Piece>(int32_t(lhs) | int32_t(rhs)); }
inline int32_t operator<<(Piece p, int32_t shift) { return static_cast<int32_t>(p) << shift; }

//空のマスに対するチャンネルを用意して入力する必要はあるんだろうか？
constexpr int64_t INPUT_CHANNEL_NUM = 2;

inline Piece oppositeColor(Piece p) { return (p == BLACK_PIECE ? WHITE_PIECE : BLACK_PIECE); }

std::ostream& operator<<(std::ostream&, Piece piece);

} // namespace Go

#endif //MIACIS_GO_PIECE_HPP