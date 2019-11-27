#include"../neural_network.hpp"
#include"position.hpp"

int main()
{
    std::cout << (torch::cuda::is_available() ?
                  "CUDA is available." :
                  "CUDA is not available.") << std::endl;
    nn->setGPU(0);
    nn->eval();

    Position::initHashSeed();

    std::cout << "ここでUSI的なものを起動してなんなりとやる" << std::endl;
}