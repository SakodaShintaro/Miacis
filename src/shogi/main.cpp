#include "interface.hpp"

int main() {
    std::cout << (torch::cuda::is_available() ? "CUDA is available." : "CUDA is not available.") << std::endl;

    Position::initHashSeed();

    Bitboard::init();

    HuffmanCodedPos::init();

    Interface usi;
    usi.loop();
}