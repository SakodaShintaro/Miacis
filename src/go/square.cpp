#include "square.hpp"
#include <sstream>

namespace Go {

const Dir DirList[8] = {
    //前から時計回りに
    U, RU, R, RD, D, LD, L, LU
};

std::string squareToString(Square sq) {
    std::stringstream ss;
    int32_t x = sq % BOARD_WIDTH;
    if (x >= 8) {
        x++;
    }
    ss << (char)('A' + x) << 1 + sq / BOARD_WIDTH;
    return ss.str();
}

} // namespace Go