#include "interface.hpp"
#include "../game.hpp"
#include "../learn.hpp"
#include "test.hpp"

namespace Shogi {

Interface::Interface() : searcher_(nullptr) {
    //メンバ関数
    // clang-format off
    command_["usi"]        = [this] { usi(); };
    command_["isready"]    = [this] { isready(); };
    command_["setoption"]  = [this] { setoption(); };
    command_["usinewgame"] = [this] { usinewgame(); };
    command_["position"]   = [this] { position(); };
    command_["go"]         = [this] { go(); };
    command_["stop"]       = [this] { stop(); };
    command_["quit"]       = [this] { quit(); };
    command_["gameover"]   = [this] { gameover(); };

    //メンバ関数以外
    command_["cleanGames"]         = cleanGames;
    command_["supervisedLearn"]    = supervisedLearn;
    command_["reinforcementLearn"] = reinforcementLearn;
    command_["contrastiveLearn"]   = contrastiveLearn;
    command_["test"]               = test;
    command_["infiniteTest"]       = infiniteTest;
    command_["checkSearchSpeed"]   = checkSearchSpeed;
    command_["checkSearchSpeed2"]  = checkSearchSpeed2;
    command_["checkGenSpeed"]      = checkGenSpeed;
    command_["checkPredictSpeed"]  = checkPredictSpeed;
    command_["checkVal"]           = checkVal;
    command_["checkValInfer"]      = checkValInfer;
    command_["checkDoAndUndo"]     = checkDoAndUndo;
    command_["checkMirror"]        = checkMirror;
    command_["checkBook"]          = checkBook;
    command_["makeBook"]           = makeBook;
    command_["searchWithLog"]      = searchWithLog;
    command_["testLoad"]           = testLoad;
    command_["testDLShogiModel"]   = testDLShogiModel;
    command_["testDLShogiONNXModel"] = testDLShogiONNXModel;
    command_["checkValDLShogi"]    = checkValDLShogi;
    command_["testModel"]          = testModel;
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

void Interface::usi() {
#ifdef USE_CATEGORICAL
    std::cout << "id name Miacis_categorical" << std::endl;
#else
    std::cout << "id name Miacis_scalar" << std::endl;
#endif
    std::cout << "id author Sakoda Shintaro" << std::endl;

    for (const auto& pair : search_options_.check_options) {
        std::cout << "option name " << pair.first << " type check default " << std::boolalpha << pair.second.value << std::endl;
    }
    for (const auto& pair : search_options_.spin_options) {
        std::cout << "option name " << pair.first << " type spin default " << pair.second.value << " min " << pair.second.min
                  << " max " << pair.second.max << std::endl;
    }
    for (const auto& pair : search_options_.filename_options) {
        std::cout << "option name " << pair.first << " type filename default " << pair.second.value << std::endl;
    }

    printf("usiok\n");
}

void Interface::isready() {
    searcher_ = std::make_unique<SearcherForPlay>(search_options_);
    printf("readyok\n");
}

void Interface::setoption() {
    std::string input;
    std::cin >> input;
    assert(input == "name");
    std::cin >> input;

    for (auto& pair : search_options_.check_options) {
        if (input == pair.first) {
            std::cin >> input;
            std::cin >> input;
            pair.second.value = (input == "true");
            return;
        }
    }
    for (auto& pair : search_options_.spin_options) {
        if (input == pair.first) {
            std::cin >> input;
            std::cin >> pair.second.value;
            return;
        }
    }
    for (auto& pair : search_options_.filename_options) {
        if (input == pair.first) {
            std::cin >> input;
            std::cin >> pair.second.value;
            return;
        }
    }
}

void Interface::usinewgame() {}

void Interface::position() {
    //Ponderが走っているかもしれないので一度止める
    //root_をthinkに参照で与えているのでposition構築前に止める必要がある
    //値渡しにすれば大丈夫だろうけど、別に棋力的に大差はないだろう
    searcher_->stop_signal = true;
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
    root_.fromStr(sfen);

    std::cin >> input; //input == "moves" or "go"となる
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

void Interface::go() {
    searcher_->stop_signal = false;

    int64_t time_limit = 0;
    std::string input;
    std::cin >> input;
    if (input == "ponder") {
        //ponderの処理
        //指し手を指定してのponderは行わないのでここには来ない
    } else if (input == "btime") {
        std::cin >> input;
        int64_t btime = stoll(input);
        std::cin >> input; //input == "wtime" となるはず
        std::cin >> input;
        int64_t wtime = stoll(input);
        int64_t time = (root_.color() == BLACK ? btime : wtime);
        int64_t remained_turn = (search_options_.draw_turn - root_.turnNumber()) / 2;
        remained_turn = (remained_turn + search_options_.remained_turn_divisor - 1) / search_options_.remained_turn_divisor;
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
        search_options_.random_turn = 0;
    } else if (input == "mate") {
        //詰み探索(未実装)
        std::cout << "詰み探索は未実装" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(10));
        std::exit(1);
    }

    //思考開始
    //別スレッドで思考させてこのスレッドはコマンド受付ループに戻る
    //Ponderオンの場合、一度与えられた持ち時間でbestmoveを弾き出したあと無限の持ち時間で現局面についてPonderを行う
    //予想手を決めなくとも置換表を埋めていくだけで強くなるはず
    thread_ = std::thread([this, time_limit]() {
        Move best_move =
            (root_.canWinDeclare() ? DECLARE_MOVE : searcher_->think(root_, time_limit - search_options_.byoyomi_margin));
        std::cout << "bestmove " << best_move << std::endl;
        if (search_options_.USI_Ponder && best_move != NULL_MOVE && best_move != DECLARE_MOVE) {
            root_.doMove(best_move);
            float score{};
            if (!root_.isFinish(score) && root_.turnNumber() <= search_options_.draw_turn) {
                searcher_->think(root_, LLONG_MAX);
            }
        }
    });
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

void Interface::gameover() {
    stop();
    std::string input;
    std::cin >> input;
    //"win" or "lose" or "draw" が来るらしいが、特にするべきことが見当たらない
}

} // namespace Shogi