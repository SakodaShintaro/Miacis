#include "square.hpp"
#include <sstream>

namespace Go {

const Dir DirList[8] = {
    //前から時計回りに
    U, RU, R, RD, D, LD, L, LU
};

std::string squareToString(Square sq) {
    std::stringstream ss;
    ss << (char)('A' + sq % BOARD_WIDTH) << 1 + sq / BOARD_WIDTH;
    return ss.str();
}

} // namespace Go