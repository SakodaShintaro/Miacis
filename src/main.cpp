#include"usi.hpp"
#include"test.hpp"
#include"neural_network.hpp"

int main()
{
    std::cout << (torch::cuda::is_available() ?
                  "CUDA is available." :
                  "CUDA is not available.") << std::endl;
    nn->setGPU(0);
    nn->eval();

    ShogiPosition::initHashSeed();

    Bitboard::init();

    USI usi;
    usi.loop();
}