#include"usi.hpp"
#include"usi_options.hpp"
#include"test.hpp"
#include"bitboard.hpp"
#include"MCTSearcher.hpp"
#include"neural_network.hpp"

std::unique_ptr<NeuralNetwork<Tensor>> nn(new NeuralNetwork<Tensor>);

int main()
{
    //devices::Naive dev;
    devices::CUDA dev(0);
    Device::set_default(dev);
    Graph g;
    Graph::set_default(g);

    nn->init();
    nn->save(MODEL_PATH);

    initCanMove();

    Position::initHashSeed();

    Bitboard::init();

    USI usi;
    usi.loop();
}