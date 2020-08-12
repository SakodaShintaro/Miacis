#include "usi.hpp"

int main() {
    std::cout << (torch::cuda::is_available() ? "CUDA is available." : "CUDA is not available.") << std::endl;

    Position::initHashSeed();

    Bitboard::init();

    USI usi;
    usi.loop();
}