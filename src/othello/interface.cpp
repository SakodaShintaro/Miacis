#include"interface.hpp"
#include"../game.hpp"
#include"../neural_network.hpp"
#include"../learn.hpp"

Interface::Interface() : searcher_(nullptr) {
    //メンバ関数
    command_["printOption"] = std::bind(&Interface::printOption, this);
    command_["setoption"]   = std::bind(&Interface::setoption,   this);
    command_["play"]        = std::bind(&Interface::play,        this);
    command_["stop"]        = std::bind(&Interface::stop,        this);
    command_["quit"]        = std::bind(&Interface::quit,        this);

    //メンバ関数以外
    command_["initParams"]         = initParams;
    command_["searchLearningRate"] = searchLearningRate;
    command_["alphaZero"]          = alphaZero;
}

void Interface::loop() {
    std::string input;
    while (std::cin >> input) {
        if (command_.count(input)) {
            command_[input]();
        } else {
            std::cout << "Illegal input: " << input << std::endl;
        }
    }
    quit();
}

void Interface::printOption() {
#ifdef USE_CATEGORICAL
    std::cout << "id name Miacis_categorical" << std::endl;
#else
    std::cout << "id name Miacis_scalar" << std::endl;
#endif
    std::cout << "id author Sakoda Shintaro" << std::endl;

    for (const auto& pair : options_.check_options) {
        std::cout << "option name " << pair.first << " type check default " << std::boolalpha << pair.second.value << std::endl;
    }
    for (const auto& pair : options_.spin_options) {
        std::cout << "option name " << pair.first << " type spin default " << pair.second.value
                  << " min " << pair.second.min << " max " << pair.second.max << std::endl;
    }
    for (const auto& pair : options_.filename_options) {
        std::cout << "option name " << pair.first << " type filename default " << pair.second.value << std::endl;
    }

    printf("usiok\n");
}

void Interface::setoption() {
    std::string input;
    std::cin >> input;
    assert(input == "name");
    std::cin >> input;

    for (auto& pair : options_.check_options) {
        if (input == pair.first) {
            std::cin >> input;
            std::cin >> input;
            pair.second.value = (input == "true");
            return;
        }
    }
    for (auto& pair : options_.spin_options) {
        if (input == pair.first) {
            std::cin >> input;
            std::cin >> pair.second.value;
            return;
        }
    }
    for (auto& pair : options_.filename_options) {
        if (input == pair.first) {
            std::cin >> input;
            std::cin >> pair.second.value;
            return;
        }
    }
}

void Interface::play() {
    torch::load(nn, options_.model_name);
    nn->setGPU(0);
    printf("readyok\n");
    searcher_ = std::make_unique<SearcherForPlay>(options_, nn);

    while (!root_.isFinish()) {
        root_.print();

        Move best_move = searcher_->think(root_, options_.byoyomi_margin);
        std::cout << "bestmove " << best_move << std::endl;

        root_.doMove(best_move);
    }
}

void Interface::stop() {
    Searcher::stop_signal = true;
    if (thread_.joinable()) {
        thread_.join();
    }
}

void Interface::quit() {
    stop();
    exit(0);
}

