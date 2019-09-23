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
    command_["quit"]       = std::bind(&USI::quit,       this);
    command_["gameover"]   = std::bind(&USI::gameover,   this);

    //メンバ関数以外
    command_["initParams"]         = initParams;
    command_["cleanGames"]         = cleanGames;
    command_["searchLearningRate"] = searchLearningRate;
    command_["supervisedLearn"] = supervisedLearn;
    command_["alphaZero"] = alphaZero;
    command_["test"] = test;
    command_["checkSearchSpeed"] = checkSearchSpeed;
    command_["checkGenSpeed"] = checkGenSpeed;
    command_["checkPredictSpeed"] = checkPredictSpeed;
    command_["checkVal"] = checkVal;
    command_["checkTransitionModel"] = checkTransitionModel;
    command_["checkActionRepresentations"] = checkActionRepresentations;
}

void USI::loop() {
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

void USI::usi() {
#ifdef USE_CATEGORICAL
    std::cout << "id name Miacis_categorical" << std::endl;
#else
    std::cout << "id name Miacis_scalar" << std::endl;
#endif
    std::cout << "id author Sakoda Shintaro" << std::endl;

    for (const auto& pair : usi_options_.check_options) {
        std::cout << "option name " << pair.first << " type check default " << std::boolalpha << pair.second.value << std::endl;
    }
    for (const auto& pair : usi_options_.spin_options) {
        std::cout << "option name " << pair.first << " type spin default " << pair.second.value
                  << " min " << pair.second.min << " max " << pair.second.max << std::endl;
    }
    for (const auto& pair : usi_options_.filename_options) {
        std::cout << "option name " << pair.first << " type filename default " << pair.second.value << std::endl;
    }

    printf("usiok\n");
}

void USI::isready() {
    torch::load(nn, usi_options_.model_name);
    nn->setGPU(0);
    printf("readyok\n");
}

void USI::setoption() {
    std::string input;
    std::cin >> input;
    assert(input == "name");
    std::cin >> input;

    for (auto& pair : usi_options_.check_options) {
        if (input == pair.first) {
            std::cin >> input;
            std::cin >> input;
            pair.second.value = (input == "true");
            return;
        }
    }
    for (auto& pair : usi_options_.spin_options) {
        if (input == pair.first) {
            std::cin >> input;
            std::cin >> pair.second.value;
            return;
        }
    }
    for (auto& pair : usi_options_.filename_options) {
        if (input == pair.first) {
            std::cin >> input;
            std::cin >> pair.second.value;
            return;
        }
    }
}

void USI::usinewgame() {
    if (usi_options_.use_sim_net) {
        searcher_using_sim_net_ = std::make_unique<SearcherUsingSimNet>(usi_options_, nn);
    } else {
        searcher_ = std::make_unique<SearcherForPlay>(usi_options_, nn);
    }
}

void USI::position() {
    //Ponderが走っているかもしれないので一度止める
    //root_をthinkに参照で与えているのでposition構築前に止める必要がある
    //値渡しにすれば大丈夫だろうけど、別に棋力的に大差はないだろう
    Searcher::stop_signal = true;
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
            sfen += input + (i < 3 ? " " : "");
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
        //指し手を指定してのponderは行わないのでここには来ない
        assert(false);
    } else if (input == "btime") {
        std::cin >> input;
        int64_t btime = stoll(input);
        std::cin >> input; //input == "wtime" となるはず
        std::cin >> input;
        int64_t wtime = stoll(input);
        int64_t time = (root_.color() == BLACK ? btime : wtime);
        int64_t remained_turn = (usi_options_.draw_turn - root_.turnNumber()) / 2;
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
        usi_options_.random_turn = 0;
    } else if (input == "mate") {
        //詰み探索(未実装)
        assert(false);
    }

    //思考開始
    //別スレッドで思考させてこのスレッドはコマンド受付ループに戻る
    //Ponderオンの場合、一度与えられた持ち時間でbestmoveを弾き出したあと無限の持ち時間で現局面についてPonderを行う
    //予想手を決めなくとも置換表を埋めていくだけで強くなるはず
    thread_ = std::thread([this, time_limit]() {
        Move best_move = (usi_options_.use_sim_net ? searcher_using_sim_net_->thinkMCTS(root_, usi_options_.random_turn)
                                                   : searcher_->think(root_, time_limit - usi_options_.byoyomi_margin));
        std::cout << "bestmove " << best_move << std::endl;
        if (usi_options_.USI_Ponder && !usi_options_.use_sim_net) {
            searcher_->think(root_, LLONG_MAX);
        }
    });
}

void USI::stop() {
    Searcher::stop_signal = true;
    if (thread_.joinable()) {
        thread_.join();
    }
}

void USI::quit() {
    stop();
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
