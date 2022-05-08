#include "interface.hpp"

int main() {
    int gpu_num;
    cudaGetDeviceCount(&gpu_num);
    std::cout << "CUDA available gpu_num: " << gpu_num << std::endl;

    Position::initHashSeed();

    Bitboard::init();

    HuffmanCodedPos::init();

    Interface usi;
    usi.loop();
}