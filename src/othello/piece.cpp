#include "piece.hpp"

namespace Othello {

const ArrayMap<std::string, PieceNum> PieceToStr({ { BLACK_PIECE, "黒" }, { WHITE_PIECE, "白" }, { EMPTY, "無" } });

const ArrayMap<std::string, PieceNum> PieceToSfenStr({ { BLACK_PIECE, "x" }, { WHITE_PIECE, "o" }, { EMPTY, " " } });

std::ostream& operator<<(std::ostream& os, const Piece piece) {
    os << PieceToStr[piece];
    return os;
}

} // namespace Othello