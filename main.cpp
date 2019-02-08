#include"usi.hpp"
#include"usi_options.hpp"
#include"test.hpp"
#include"bitboard.hpp"
#include"MCTSearcher.hpp"
#include"neural_network.hpp"

#ifdef USE_LIBTORCH
NeuralNetwork nn;
#else
std::unique_ptr<NeuralNetwork<Tensor>> nn(new NeuralNetwork<Tensor>);
#endif

int main()
{
#ifdef USE_LIBTORCH
    nn->to(device);
#else
    //devices::Naive dev;
    devices::CUDA dev(0);
    Device::set_default(dev);

    nn->init();
#endif

    initCanMove();

    Position::initHashSeed();

    Bitboard::init();

    USI usi;
    usi.loop();
}