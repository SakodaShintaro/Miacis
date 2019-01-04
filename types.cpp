#include"types.hpp"

#ifndef USE_NN
//Score
std::ostream& operator<<(std::ostream& os, const Score s) {
    os << static_cast<int>(s);
    return os;
}
#endif

//Depth
std::ostream& operator<<(std::ostream& os, const Depth d) {
    os << static_cast<int>(d);
    return os;
}
std::istream& operator>>(std::istream& is, Depth& d) {
    int tmp;
    is >> tmp;
    d = Depth(tmp);
    return is;
}