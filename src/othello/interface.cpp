#include "interface.hpp"
#include "../game.hpp"
#include "../learn.hpp"
#include "../neural_network.hpp"
#include "../search_nn/learn_search_nn.hpp"
#include "../search_nn/models.hpp"
#include "../search_nn/simple_MLP/learn_simple_mlp.hpp"

namespace Othello {

Interface::Interface() : searcher_(nullptr) {
    //メンバ関数
    // clang-format off
    command_["printOption"]       = [this] { printOption(); };
    command_["set"]               = [this] { set(); };
    command_["think"]             = [this] { think(); };
    command_["test"]              = [this] { test(); };
    command_["infiniteTest"]      = [this] { infiniteTest(); };
    command_["battle"]            = [this] { battle(); };
    command_["battleVSRandom"]    = [this] { battleVSRandom(); };
    command_["outputValue"]       = [this] { outputValue(); };
    command_["init"]              = [this] { init(); };
    command_["play"]              = [this] { play(); };
    command_["go"]                = [this] { go(); };
    command_["stop"]              = [this] { stop(); };
    command_["quit"]              = [this] { quit(); };
    command_["testMCTSNet"]       = [this] { testSearchNN<MCTSNet>(); };
    command_["testProposedModel"] = [this] { testSearchNN<ProposedModel>(); };
    command_["testStackedLSTM"]   = [this] { testSearchNN<StackedLSTM>(); };
    command_["testSimpleMLP"]     = [this] { testSearchNN<SimpleMLP>(); };

    //メンバ関数以外
    command_["initParams"]        = initParams;
    command_["supervisedLearn"]   = supervisedLearn;
    command_["alphaZero"]         = alphaZero;
    command_["pretrain"]          = pretrainSimpleMLP;
    command_["learnMCTSNet"]      = [](){ learnSearchNN<MCTSNet>("mcts_net"); };
    command_["learnStackedLSTM"]  = [](){ learnSearchNN<StackedLSTM>("stacked_lstm"); };
    command_["validMCTSNet"]      = [](){ validSearchNN<MCTSNet>("mcts_net"); };
    // clang-format on
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
        std::cout << "option name " << pair.first << " type spin default " << pair.second.value << " min " << pair.second.min
                  << " max " << pair.second.max << std::endl;
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
    SearchOptions search_options;
    search_options.search_limit = 800;
    search_options.print_interval = 100000;
    search_options.thread_num_per_gpu = 1;
    search_options.search_batch_size = 1;
    search_options.output_log_file = true;
    NeuralNetwork nn;
    torch::load(nn, NeuralNetworkImpl::DEFAULT_MODEL_NAME);
    nn->setGPU(0);
    nn->eval();
    SearcherForPlay searcher(search_options);

    Position pos;
    Game game;

    auto start = std::chrono::steady_clock::now();
    while (true) {
        Move best_move = searcher.think(pos, LLONG_MAX);

        if (best_move == NULL_MOVE) {
            //投了
            game.result = (pos.color() == BLACK ? MIN_SCORE : MAX_SCORE);
            break;
        }

        float finish_score{};
        if ((pos.isFinish(finish_score) && finish_score == (MAX_SCORE + MIN_SCORE) / 2) ||
            pos.turnNumber() > search_options.draw_turn) {
            //千日手or持将棋
            game.result = finish_score;
            break;
        }

        pos.doMove(best_move);
        OneTurnElement element;
        element.move = best_move;
        game.elements.push_back(element);
    }
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << elapsed.count() / pos.turnNumber() << " msec / pos" << std::endl;

    game.writeKifuFile("./");
    std::cout << "finish test" << std::endl;
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
            float score{};
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
    int64_t turn = 0;
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
        float score{};
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

    for (int64_t i = 0; i < 100; i++) {
        root_.init();

        while (true) {
            float score{};
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
    if (options_.use_mcts_net) {
        mcts_net_ = MCTSNet(options_);
        torch::load(mcts_net_, options_.model_name);
        mcts_net_->eval();
    } else if (options_.use_stacked_lstm) {
        stacked_lstm_ = StackedLSTM(options_);
        torch::load(stacked_lstm_, options_.model_name);
        stacked_lstm_->eval();
    } else {
        searcher_ = std::make_unique<SearcherForPlay>(options_);
    }
}

void Interface::play() {
    std::string input;
    std::cin >> input;
    Move move = stringToMove(input);
    root_.doMove(move);
}

void Interface::go() {
    if (options_.use_mcts_net) {
        torch::NoGradGuard no_grad_guard;
        Move best_move = mcts_net_->think(root_, options_.byoyomi_margin);
        std::cout << "best_move " << best_move << std::endl;
        root_.doMove(best_move);
    } else if (options_.use_stacked_lstm) {
        torch::NoGradGuard no_grad_guard;
        Move best_move = stacked_lstm_->think(root_, options_.byoyomi_margin);
        std::cout << "best_move " << best_move << std::endl;
        root_.doMove(best_move);
    } else {
        Move best_move = searcher_->think(root_, options_.byoyomi_margin);
        std::cout << "best_move " << best_move << std::endl;
        root_.doMove(best_move);
    }
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

    std::uniform_real_distribution<float> dist(0.0, 1.0);

    float score{};
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
        for (; index < moves.size(); index++) {
            if ((probability_sum += policy[index]) >= threshold) {
                break;
            }
        }
        if (root_.turnNumber() >= 10) {
            index = std::max_element(policy.begin(), policy.end()) - policy.begin();
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
    root_.print();
    std::cout << score << std::endl;
    ofs << -1 << std::endl;
    std::cout << "finish outputValue" << std::endl;
}

template<class T> void Interface::testSearchNN() {
    root_.init();

    options_.search_limit = 10;
    options_.print_interval = INT_MAX;
    options_.print_policy_num = 800;
    options_.search_batch_size = 1;
    options_.thread_num_per_gpu = 1;
    T model(options_);
    model->setGPU(0);
    model->eval();
    torch::NoGradGuard no_grad_guard;

    float score;
    while (!root_.isFinish(score)) {
        Move best_move = model->think(root_, INT_MAX);
        root_.doMove(best_move);
        root_.print();
    }
}

} // namespace Othello