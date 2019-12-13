#include"interface.hpp"
#include"../game.hpp"
#include"../neural_network.hpp"
#include"../learn.hpp"

Interface::Interface() : searcher_(nullptr) {
    //メンバ関数
    command_["printOption"] = std::bind(&Interface::printOption, this);
    command_["set"]         = std::bind(&Interface::set,         this);
    command_["think"]       = std::bind(&Interface::think,       this);
    command_["test"]        = std::bind(&Interface::test,        this);
    command_["battle"]      = std::bind(&Interface::battle,      this);
    command_["init"]        = std::bind(&Interface::init,        this);
    command_["play"]        = std::bind(&Interface::play,        this);
    command_["go"]          = std::bind(&Interface::go,          this);
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

void Interface::set() {
    std::string input;
    std::cin >> input;

    for (auto& pair : options_.check_options) {
        if (input == pair.first) {
            std::cin >> input;
            pair.second.value = (input == "true");
            return;
        }
    }
    for (auto& pair : options_.spin_options) {
        if (input == pair.first) {
            std::cin >> pair.second.value;
            return;
        }
    }
    for (auto& pair : options_.filename_options) {
        if (input == pair.first) {
            std::cin >> pair.second.value;
            return;
        }
    }
}

void Interface::think() {
    root_.init();

    //対局の準備
    torch::load(nn, options_.model_name);
    nn->setGPU(0);

    options_.search_limit = 80000;
    options_.print_interval = INT_MAX;
    options_.print_policy_num = 800;
    options_.search_batch_size = 1;
    options_.thread_num = 1;
    searcher_ = std::make_unique<SearcherForPlay>(options_, nn);

    searcher_->think(root_, 1000000);
}

void Interface::test() {
    root_.init();

    //対局の準備
    torch::load(nn, options_.model_name);
    nn->setGPU(0);
    options_.search_limit = 800;
    options_.search_batch_size = 1;
    options_.thread_num = 1;
    options_.print_interval = INT_MAX;
    options_.print_policy_num = 800;
    searcher_ = std::make_unique<SearcherForPlay>(options_, nn);

    while (true) {
        float score;
        root_.print();
        if (root_.isFinish(score)) {
            std::cout << "score = " << score << std::endl;
            break;
        }

        Move best_move = searcher_->think(root_, 1000000);
        root_.doMove(best_move);
    }
}

void Interface::battle() {
    root_.init();

    //手番の入力
    std::cout << "人間の手番(0 or 1): ";
    int64_t turn;
    std::cin >> turn;

    //対局の準備
    torch::load(nn, options_.model_name);
    nn->setGPU(0);
    options_.search_limit = 800;
    options_.search_batch_size = 1;
    options_.thread_num = 1;
    options_.print_interval = INT_MAX;
    options_.print_policy_num = 800;
    searcher_ = std::make_unique<SearcherForPlay>(options_, nn);

    while (true) {
        float score;
        root_.print();
        if (root_.isFinish(score)) {
            std::cout << "score = " << score << std::endl;
            break;
        }

        if (root_.turnNumber() % 2 == turn) {
            while (true) {
                std::string input;
                std::cin >> input;
                Move move = stringToMove(input);
                if (root_.isLegalMove(move)) {
                    root_.doMove(move);
                    break;
                } else {
                    std::cout << "非合法手" << std::endl;
                }
            }
        } else {
            Move best_move = searcher_->think(root_, 1000000);
            root_.doMove(best_move);
        }
    }
}

void Interface::init() {
    root_.init();

    //対局の準備
    torch::load(nn, options_.model_name);
    nn->setGPU(0);
    searcher_ = std::make_unique<SearcherForPlay>(options_, nn);
}

void Interface::play() {
    std::string input;
    std::cin >> input;
    Move move = stringToMove(input);
    root_.doMove(move);
}

void Interface::go() {
    Move best_move = searcher_->think(root_, options_.byoyomi_margin);
    std::cout << "best_move " << best_move << std::endl;
    root_.doMove(best_move);
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