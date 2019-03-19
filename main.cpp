#include"usi.hpp"
#include"usi_options.hpp"
#include"test.hpp"
#include"bitboard.hpp"
#include"neural_network.hpp"

int main()
{
    nn->setGPU(0);
    nn->eval();

    Position::initHashSeed();

    Bitboard::init();

    USI usi;
    usi.loop();
}