#include"usi.hpp"
#include"usi_options.hpp"
#include"test.hpp"
#include"bitboard.hpp"
#include"neural_network.hpp"

int main()
{
#ifdef USE_LIBTORCH
    nn->setGPU(0);
    nn->eval();
#else
    //devices::Naive dev;
    devices::CUDA dev(0);
    Device::set_default(dev);

    nn->init();
#endif

    Position::initHashSeed();

    Bitboard::init();

    USI usi;
    usi.loop();
}