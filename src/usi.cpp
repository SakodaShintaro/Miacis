#include"usi.hpp"
#include"game.hpp"
#include"test.hpp"
#include"neural_network.hpp"
#include"learn.hpp"

USI::USI() : searcher_(nullptr) {
    //メンバ関数
    command_["usi"]        = std::bind(&USI::usi,        this);
    command_["isready"]    = std::bind(&USI::isready,    this);
    command_["setoption"]  = std::bind(&USI::setoption,  this);
    command_["usinewgame"] = std::bind(&USI::usinewgame, this);
    command_["position"]   = std::bind(&USI::position,   this);
    command_["go"]         = std::bind(&USI::go,         this);
    command_["stop"]       = std::bind(&USI::stop,       this);
    command_["ponderhit"]  = std::bind(&USI::ponderhit,  this);
    command_["quit"]       = std::bind(&USI::quit,       this);
    command_["gameover"]   = std::bind(&USI::gameover,   this);

    //メンバ関数以外
    command_["initParams"]         = initParams;
    command_["cleanGames"]         = cleanGames;
    command_["searchLearningRate"] = searchLearningRate;
    command_["supervisedLearn"]    = supervisedLearn;
    command_["alphaZero"]          = alphaZero;
    command_["test"]               = test;
    command_["checkSearchSpeed"]   = checkSearchSpeed;
    command_["checkGenSpeed"]      = checkGenSpeed;
    command_["checkPredictSpeed"]  = checkPredictSpeed;
    command_["checkVal"]           = checkVal;
}

void USI::loop() {
    std::string input;
    while (std::cin >> input) {
        if (command_.count(input)) {
            command_[input]();
        } else {
            std::cout << "Illegal input" << std::endl;
        }
    }
}

void USI::usi() {
#ifdef USE_CATEGORICAL
    std::cout << "id name Miacis_categorical" << std::endl;
#else
    std::cout << "id name Miacis_scalar" << std::endl;
#endif
    std::cout << "id author Sakoda Shintaro" << std::endl;

    usi_option_.byoyomi_margin = 0;
    std::cout << "option name byoyomi_margin type spin default 0 min 0 max 10000" << std::endl;

    usi_option_.random_turn = 0;
    std::cout << "option name random_turn type spin default 0 min 0 max 1000" << std::endl;

    usi_option_.thread_num = 2;
    std::cout << "option name thread_num type spin default 2 min 1 max 2048" << std::endl;

    usi_option_.search_batch_size = 128;
    std::cout << "option name search_batch_size type spin default 128 min 1 max 2048" << std::endl;

    usi_option_.draw_turn = 256;
    std::cout << "option name draw_turn type spin default 256 min 0 max 4096" << std::endl;

    usi_option_.print_policy_num = 0;
    std::cout << "option name print_policy_num type spin default 0 min 0 max 593" << std::endl;

    usi_option_.print_interval = 10000;
    std::cout << "option name print_interval type spin default 10000 min 1 max 100000000" << std::endl;

    usi_option_.search_limit = 1000000000;
    std::cout << "option name search_limit type spin default 1000000000 min 1 max 1000000000" << std::endl;

    usi_option_.C_PUCT_x1000 = 2500;
    std::cout << "option name C_PUCT_x1000 type spin default 2500 min 0 max 1000000" << std::endl;

    usi_option_.temperature_x1000 = 0;
    std::cout << "option name temperature_x1000 type spin default 0 min 0 max 1000000000" << std::endl;

    usi_option_.UCT_lambda_x1000 = 1000;
    std::cout << "option name UCT_lambda_x1000 type spin default 1000 min 0 max 1000" << std::endl;

    usi_option_.model_name = NeuralNetworkImpl::DEFAULT_MODEL_NAME;
    std::cout << "option name model_name type string default " << usi_option_.model_name << std::endl;

    usi_option_.USI_Hash = 256;
    printf("usiok\n");
}

void USI::isready() {
    torch::load(nn, usi_option_.model_name);
    nn->setGPU(0);
    printf("readyok\n");
}

void USI::setoption() {
    std::string input;
    std::cin >> input;
    assert(input == "name");
    std::cin >> input;
    if (input == "byoyomi_margin") {
        std::cin >> input; //input == "value"となるなず
        std::cin >> usi_option_.byoyomi_margin;
    } else if (input == "random_turn") {
        std::cin >> input; //input == "value"となるなず
        std::cin >> usi_option_.random_turn;
    } else if (input == "USI_Hash") {
        std::cin >> input; //input == "value"となるはず
        std::cin >> usi_option_.USI_Hash;
    } else if (input == "USI_Ponder") {
        std::cin >> input; //input == "value"となるなず
        std::cin >> input; //特になにもしていない
    } else if (input == "thread_num") {
        std::cin >> input; //input == "value"となるはず
        std::cin >> usi_option_.thread_num;
    } else if (input == "search_batch_size") {
        std::cin >> input; //input == "value"となるはず
        std::cin >> usi_option_.search_batch_size;
    } else if (input == "draw_turn") {
        std::cin >> input; //input == "value"となるはず
        std::cin >> usi_option_.draw_turn;
    } else if (input == "search_limit") {
        std::cin >> input;
        std::cin >> usi_option_.search_limit;
    } else if (input == "C_PUCT_x1000") {
        std::cin >> input;
        std::cin >> usi_option_.C_PUCT_x1000;
    } else if (input == "temperature_x1000") {
        std::cin >> input;
        std::cin >> usi_option_.temperature_x1000;
    } else if (input == "UCT_lambda_x1000") {
        std::cin >> input;
        std::cin >> usi_option_.UCT_lambda_x1000;
    } else if (input == "print_policy_num") {
        std::cin >> input;
        std::cin >> usi_option_.print_policy_num;
    } else if (input == "print_interval") {
        std::cin >> input;
        std::cin >> usi_option_.print_interval;
    } else if (input == "model_name") {
        std::cin >> input;
        std::cin >> usi_option_.model_name;
    }
}

void USI::usinewgame() {
    searcher_ = std::make_unique<SearcherForPlay>(usi_option_.USI_Hash * 1024 * 1024 / 20000,
                                                  usi_option_.C_PUCT_x1000 / 1000.0,
                                                  usi_option_.thread_num,
                                                  usi_option_.search_batch_size,
                                                  nn,
                                                  usi_option_.temperature_x1000 / 1000.0,
                                                  usi_option_.UCT_lambda_x1000 / 1000.0,
                                                  usi_option_.print_policy_num,
                                                  usi_option_.draw_turn);
}

void USI::position() {
    if (thread_.joinable()) {
        thread_.join();
    }

    //局面の構築
    std::string input, sfen;
    std::cin >> input;
    if (input == "startpos") {
        sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
    } else {
        for (int i = 0; i < 4; i++) {
            std::cin >> input;
            sfen += input;
            sfen += " ";
        }
    }
    root_.init();
    root_.loadSFEN(sfen);

    std::cin >> input;  //input == "moves" or "go"となる
    if (input != "go") {
        while (std::cin >> input) {
            if (input == "go") {
                break;
            }
            //inputをMoveに直して局面を動かす
            Move move = stringToMove(input);
            move = root_.transformValidMove(move);
            root_.doMove(move);
        }
    }

    go();
}

void USI::go() {
    Searcher::stop_signal = false;

    int64_t time_limit;
    std::string input;
    std::cin >> input;
    if (input == "ponder") {
        //ponderの処理
    } else if (input == "btime") {
        std::cin >> input;
        int64_t btime = stoll(input);
        std::cin >> input; //input == "wtime" となるはず
        std::cin >> input;
        int64_t wtime = stoll(input);
        int64_t time = (root_.color() == BLACK ? btime : wtime);
        int64_t remained_turn = (usi_option_.draw_turn - root_.turnNumber()) / 2;
        int64_t curr_time = (remained_turn == 0 ? 0 : time / remained_turn);
        std::cin >> input; //input == "byoyomi" or "binc"となるはず
        if (input == "byoyomi") {
            std::cin >> input;
            time_limit = stoll(input) + curr_time;
        } else {
            std::cin >> input;
            int64_t binc = stoll(input);
            std::cin >> input; //input == "winc" となるはず
            std::cin >> input;
            //wincは使わないので警告が出る.鬱陶しいのでコメントアウト
            //int64_t winc = stoll(input);
            time_limit = binc + curr_time;
        }
    } else if (input == "infinite") {
        //思考時間をほぼ無限に
        time_limit = LLONG_MAX;

        //random_turnをなくす
        usi_option_.random_turn = 0;
    } else if (input == "mate") {
        //詰み探索(未実装)
        assert(false);
    }

    //思考開始
    //thinkを直接書くとstopコマンドを受け付けられなくなってしまうので別スレッドに投げる
    thread_ = std::thread([this, time_limit]() {
        auto best_move = searcher_->think(root_,
                                          time_limit - usi_option_.byoyomi_margin,
                                          usi_option_.search_limit,
                                          usi_option_.random_turn,
                                          usi_option_.print_interval);
        std::cout << "bestmove " << best_move << std::endl;
    });
}

void USI::stop() {
    Searcher::stop_signal = true;
    thread_.join();
}

void USI::ponderhit() {
    //まだ未実装
}

void USI::quit() {
    exit(0);
}

void USI::gameover() {
    std::string input;
    std::cin >> input;
    if (input == "win") {
        //勝ち
    } else if (input == "lose") {
        //負け
    } else if (input == "draw") {
        //引き分け
        return;
    }
}