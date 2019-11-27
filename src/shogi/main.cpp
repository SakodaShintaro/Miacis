#include"usi.hpp"
#include"../neural_network.hpp"

int main()
{
    std::cout << (torch::cuda::is_available() ?
                  "CUDA is available." :
                  "CUDA is not available.") << std::endl;
    nn->setGPU(0);
    nn->eval();

    Position::initHashSeed();

    Bitboard::init();

    USI usi;
    usi.loop();
}