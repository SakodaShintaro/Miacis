#include"usi.hpp"
#include"move.hpp"
#include"position.hpp"
#include"usi_options.hpp"
#include"game.hpp"
#include"test.hpp"
#include"neural_network.hpp"
#include"operate_params.hpp"
#include"learn.hpp"
#include "searcher_for_play.hpp"
#include<iostream>
#include<string>
#include<climits>
#include<thread>

USIOption usi_option;

void USI::loop() {
    std::string input;
    while (std::cin >> input) {
        if (input == "usi") {
            usi();
        } else if (input == "isready") {
            isready();
        } else if (input == "setoption") {
            setoption();
        } else if (input == "usinewgame") {
            usinewgame();
        } else if (input == "position") {
            position();
        } else if (input == "go") {
            go();
        } else if (input == "stop") {
            stop();
        } else if (input == "ponderhit") {
            ponderhit();
        } else if (input == "quit") {
            quit();
        } else if (input == "gameover") {
            gameover();
        } else if (input == "prepareForLearn") {
            torch::save(nn, MODEL_PATH);
            std::cout << "初期化したパラメータを出力" << std::endl;
        } else if (input == "cleanGame") {
            std::cout << "棋譜のあるフォルダへのパス : ";
            std::string file_path;
            std::cin >> file_path;
            cleanGames(file_path);
        } else if (input == "supervisedLearn") {
            supervisedLearn();
        } else if (input == "alphaZero") {
            alphaZero();
        } else if (input == "test") {
            test();
        } else if (input == "checkSearchSpeed") {
            checkSearchSpeed();
        } else if (input == "checkGenSpeed") {
            checkGenSpeed();
        } else if (input == "checkVal") {
            checkVal();
        } else {
            std::cout << "Illegal input" << std::endl;
        }
    }
}

void USI::usi() {
#ifdef USE_CATEGORICAL
    printf("id name TorchCa\n");
#else
    printf("id name TorchSa\n");
#endif
    printf("id author Sakoda Shintaro\n");
	printf("option name byoyomi_margin type spin default 0 min 0 max 1000\n");
    usi_option.byoyomi_margin = 0;
	printf("option name random_turn type spin default 0 min 0 max 1000\n");
    usi_option.random_turn = 0;
    printf("option name thread_num type spin default 1 min 1 max 2048\n");
    usi_option.thread_num = 1;
    printf("option name search_batch_size type spin default 1 min 1 max 2048\n");
    usi_option.search_batch_size = 1;
    printf("option name draw_turn type spin default 256 min 0 max 4096\n");
    usi_option.draw_turn = 256;

    auto d = (unsigned long long)1e9;
    printf("option name search_limit type spin default %llu min 1 max %llu\n", d, d);
    usi_option.search_limit = (int64_t)d;

    usi_option.USI_Hash = 256;
	printf("usiok\n");
}

void USI::isready() {
    torch::load(nn, MODEL_PATH);
    printf("readyok\n");
}

void USI::setoption() {
    std::string input;
    while (true) {
        std::cin >> input;
        if (input == "name") {
            std::cin >> input;
            //ここで処理
            if (input == "byoyomi_margin") {
                std::cin >> input; //input == "value"となるなず
                std::cin >> usi_option.byoyomi_margin;
                return;
            } else if (input == "random_turn") {
                std::cin >> input; //input == "value"となるなず
                std::cin >> usi_option.random_turn;
                return;
            } else if (input == "USI_Hash") {
                std::cin >> input; //input == "value"となるはず
                std::cin >> usi_option.USI_Hash;
                return;
            } else if (input == "USI_Ponder") {
                std::cin >> input; //input == "value"となるなず
                std::cin >> input; //特になにもしていない
                return;
            } else if (input == "thread_num") {
                std::cin >> input; //input == "value"となるはず
                std::cin >> usi_option.thread_num;
                return;
            } else if (input == "search_batch_size") {
                std::cin >> input; //input == "value"となるはず
                std::cin >> usi_option.search_batch_size;
                return;
            } else if (input == "draw_turn") {
                std::cin >> input; //input == "value"となるはず
                std::cin >> usi_option.draw_turn;
                return;
            } else if (input == "search_limit") {
                std::cin >> input;
                std::cin >> usi_option.search_limit;
                return;
            }
        }
    }
}

void USI::usinewgame() {
}

void USI::position() {
    if (thread_.joinable()) {
        thread_.join();
    }

    //rootを初期化
    root_.init();

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
    usi_option.stop_signal = false;
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
        std::cin >> input; //input == "byoyomi" or "binc"となるはず
        if (input == "byoyomi") {
            std::cin >> input;
            usi_option.limit_msec = stoll(input);
        } else {
            int64_t binc, winc;
            std::cin >> input;
            binc = stoll(input);
            std::cin >> input; //input == "winc" となるはず
            std::cin >> input;
            winc = stoll(input);
            usi_option.limit_msec = binc;
        }
    } else if (input == "infinite") {
        //stop来るまで思考し続ける
        //思考時間をほぼ無限に
        usi_option.limit_msec = LLONG_MAX;
        
        //random_turnをなくす
        usi_option.random_turn = 0;
    } else if (input == "mate") {
        //詰み探索(未実装)
        assert(false);
    }

    //思考開始
    //thinkを直接書くとstopコマンドを受け付けられなくなってしまうので
    //別スレッドに投げる
    thread_ = std::thread([&]() {
        SearcherForPlay searcher(usi_option.USI_Hash * 1024 * 1024 / sizeof(UctHashEntry), usi_option.thread_num, usi_option.search_batch_size, nn);
        auto best_move = searcher.think(root_);
        if (best_move == NULL_MOVE) {
            std::cout << "bestmove resign" << std::endl;
        } else {
            std::cout << "bestmove " << best_move << std::endl;
        }
    });
}

void USI::stop() {
    usi_option.stop_signal = true;
    if (thread_.joinable()) {
        thread_.join();
    }
}

void USI::ponderhit() {
    //まだ未実装
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