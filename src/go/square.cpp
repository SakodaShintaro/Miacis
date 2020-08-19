#include "square.hpp"
#include <sstream>

namespace Go {

const Dir DirList[8] = {
    //前から時計回りに
    U, RU, R, RD, D, LD, L, LU
};

std::string squareToString(Square sq) {
    std::stringstream ss;
    ss << fileToString(sq % BOARD_WIDTH) << rankToString(sq / BOARD_WIDTH);
    return ss.str();
}

} // namespace Go