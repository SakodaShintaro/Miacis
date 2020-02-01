#include"interface.hpp"
#include"../neural_network.hpp"
#include"../learn.hpp"

Interface::Interface() : searcher_(nullptr) {
    //メンバ関数
    command_["printOption"]    = std::bind(&Interface::printOption,  this);
    command_["set"]            = std::bind(&Interface::set,          this);
    command_["think"]          = std::bind(&Interface::think,        this);
    command_["test"]           = std::bind(&Interface::test,         this);
    command_["infiniteTest"]   = std::bind(&Interface::infiniteTest, this);
    command_["battle"]         = std::bind(&Interface::battle,       this);
    command_["battleVSRandom"] = std::bind(&Interface::battleVSRandom, this);
    command_["init"]           = std::bind(&Interface::init,         this);
    command_["play"]           = std::bind(&Interface::play,         this);
    command_["go"]             = std::bind(&Interface::go,           this);
    command_["stop"]           = std::bind(&Interface::stop,         this);
    command_["quit"]           = std::bind(&Interface::quit,         this);

    //メンバ関数以外
    command_["initParams"]         = initParams;
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

    options_.search_limit = 80000;
    options_.print_interval = INT_MAX;
    options_.print_policy_num = 800;
    options_.search_batch_size = 1;
    options_.thread_num_per_gpu = 1;
    searcher_ = std::make_unique<SearcherForPlay>(options_);

    searcher_->think(root_, 1000000);
}

void Interface::test() {
    root_.init();

    //対局の準備
    options_.search_limit = 800;
    options_.search_batch_size = 1;
    options_.thread_num_per_gpu = 1;
    options_.print_interval = INT_MAX;
    options_.print_policy_num = 800;
    searcher_ = std::make_unique<SearcherForPlay>(options_);

    while (true) {
        root_.print();
        float score;
        if (root_.isFinish(score)) {
            std::cout << "score = " << score << std::endl;
            break;
        }

        Move best_move = searcher_->think(root_, 1000000);
        root_.doMove(best_move);
    }
}

void Interface::infiniteTest() {
    //対局の準備
    options_.search_limit = 400;
    options_.search_batch_size = 1;
    options_.thread_num_per_gpu = 1;
    options_.random_turn = 100;
    options_.print_interval = INT_MAX;
    options_.print_policy_num = 0;
    searcher_ = std::make_unique<SearcherForPlay>(options_);

    for (int64_t i = 0; i < LLONG_MAX; i++) {
        root_.init();
        while (true) {
            root_.print();
            float score;
            if (root_.isFinish(score)) {
                std::cout << "score = " << score << std::endl;
                break;
            }

            Move best_move = searcher_->think(root_, LLONG_MAX);
            root_.doMove(best_move);
        }
    }
}

void Interface::battle() {
    root_.init();

    //手番の入力
    std::cout << "人間の手番(0 or 1): ";
    int64_t turn;
    std::cin >> turn;

    //対局の準備
    options_.search_limit = 800;
    options_.search_batch_size = 1;
    options_.random_turn = 10;
    options_.thread_num_per_gpu = 1;
    options_.print_interval = INT_MAX;
    options_.print_policy_num = 800;
    searcher_ = std::make_unique<SearcherForPlay>(options_);

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

void Interface::battleVSRandom() {
    //対局の準備
    options_.search_limit = 800;
    options_.search_batch_size = 1;
    options_.random_turn = 10;
    options_.thread_num_per_gpu = 1;
    options_.print_interval = INT_MAX;
    options_.print_policy_num = 00;
    searcher_ = std::make_unique<SearcherForPlay>(options_);

    //乱数生成器を準備
    std::mt19937_64 engine(std::random_device{}());

    for (int64_t i = 0; i < 100; i++) {
        root_.init();

        while (true) {
            float score;
            root_.print();
            if (root_.isFinish(score)) {
                std::cout << "score = " << score << std::endl;
                break;
            }

            if ((root_.turnNumber() % 2) == (i % 2)) {
                std::vector<Move> moves = root_.generateAllMoves();
                std::uniform_int_distribution<uint64_t> dist(0, moves.size() - 1);
                uint64_t index = dist(engine);
                Move best_move = moves[index];
                root_.doMove(best_move);
            } else {
                Move best_move = searcher_->think(root_, 1000000);
                root_.doMove(best_move);
            }
        }
    }
}

void Interface::init() {
    root_.init();

    //対局の準備
    searcher_ = std::make_unique<SearcherForPlay>(options_);
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
    if (searcher_ != nullptr) {
        searcher_->stop_signal = true;
    }
    if (thread_.joinable()) {
        thread_.join();
    }
}

void Interface::quit() {
    stop();
    exit(0);
}