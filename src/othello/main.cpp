#include "interface.hpp"

using namespace Othello;

int main() {
    std::cout << (torch::cuda::is_available() ? "CUDA is available." : "CUDA is not available.") << std::endl;

    Position::initHashSeed();

    Interface interface;
    interface.loop();
}