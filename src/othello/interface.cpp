#include"interface.hpp"
#include"../neural_network.hpp"
#include"../learn.hpp"

Interface::Interface() : searcher_(nullptr) {
    //メンバ関数
    command_["printOption"]    = std::bind(&Interface::printOption,    this);
    command_["set"]            = std::bind(&Interface::set,            this);
    command_["think"]          = std::bind(&Interface::think,          this);
    command_["test"]           = std::bind(&Interface::test,           this);
    command_["infiniteTest"]   = std::bind(&Interface::infiniteTest,   this);
    command_["battle"]         = std::bind(&Interface::battle,         this);
    command_["battleVSRandom"] = std::bind(&Interface::battleVSRandom, this);
    command_["outputValue"]    = std::bind(&Interface::outputValue,    this);
    command_["init"]           = std::bind(&Interface::init,           this);
    command_["play"]           = std::bind(&Interface::play,           this);
    command_["go"]             = std::bind(&Interface::go,             this);
    command_["stop"]           = std::bind(&Interface::stop,           this);
    command_["quit"]           = std::bind(&Interface::quit,           this);

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

        std::string str = root_.toStr();
        uint32_t label = best_move.toLabel();

        for (int64_t augmentation = 0; augmentation < Position::DATA_AUGMENTATION_PATTERN_NUM; augmentation++) {
            std::cout << augmentation << std::endl;
            std::string augmented_str = root_.augmentStr(str, augmentation);
            for (int64_t i = 0; i < BOARD_WIDTH; i++) {
                for (int64_t j = 0; j < BOARD_WIDTH; j++) {
                    std::cout << augmented_str[i + (BOARD_WIDTH - 1 - j) * BOARD_WIDTH];
                }
                std::cout << std::endl;
            }
            std::cout << augmented_str.back() << std::endl;
            uint32_t augmented_label = Move::augmentLabel(label, augmentation);
            std::cout << "label = " << augmented_label << " means = (" << augmented_label % BOARD_WIDTH << ", " << augmented_label / BOARD_WIDTH << ")" << std::endl;
            std::cout << std::endl;
        }

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

void Interface::outputValue() {
    root_.init();
    std::ofstream ofs("value_output.txt");
    NeuralNetwork nn;
    torch::load(nn, options_.model_name);
    nn->setGPU(0);

    std::mt19937_64 engine(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    float score;
    while (!root_.isFinish(score)) {
        std::vector<float> feature = root_.makeFeature();
        root_.print();

        std::pair<std::vector<PolicyType>, std::vector<ValueType>> y = nn->policyAndValueBatch(feature);
        PolicyType policy;
        std::vector<Move> moves = root_.generateAllMoves();
        for (const Move& move : moves) {
            policy.push_back(y.first[0][move.toLabel()]);
        }
        policy = softmax(policy);

        uint64_t index = 0;
        float probability_sum = 0;
        float threshold = dist(engine);
        for (index = 0; index < moves.size(); index++) {
            if ((probability_sum += policy[index]) >= threshold) {
                break;
            }
        }
        Move best_move = moves[index];

        ValueType value = y.second[0];
        //ターン数とValueを出力
        ofs << root_.turnNumber() << " " << best_move << std::endl;
#ifdef USE_CATEGORICAL
        for (int64_t i = 0; i < BIN_SIZE; i++) {
            ofs << value[i] << std::endl;
        }
#else
        ofs << value << std::endl;
#endif

        root_.doMove(best_move);
    }
    ofs << -1 << std::endl;
    std::cout << "finish outputValue" << std::endl;
}