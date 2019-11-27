#include"piece.hpp"
#include<array>

const ArrayMap<std::string, PieceNum> PieceToStr({
    { BLACK_PIECE, "黒" },
    { WHITE_PIECE, "白" },
    { EMPTY,       "無" }
});

const ArrayMap<std::string, PieceNum> PieceToSfenStr({
    { BLACK_PIECE, "x" },
    { WHITE_PIECE, "o" },
    { EMPTY,       " " }
});

const std::array<Piece, 28> PieceList{
    BLACK_PIECE,
    WHITE_PIECE,
};

std::ostream& operator<<(std::ostream& os, const Piece piece) {
    os << PieceToStr[piece];
    return os;
}