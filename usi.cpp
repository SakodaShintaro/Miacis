#include"usi.hpp"
#include"move.hpp"
#include"position.hpp"
#include"searcher.hpp"
#include"usi_options.hpp"
#include"eval_params.hpp"
#include"game.hpp"
#include"bonanza_method_trainer.hpp"
#include"rootstrap_trainer.hpp"
#include"alphazero_trainer.hpp"
#include"treestrap_trainer.hpp"
#include"test.hpp"
#include"network.hpp"
#include"MCTSearcher.hpp"
#include"operate_params.hpp"
#include<iostream>
#include<string>
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
#ifdef USE_NN
            eval_params->initRandom();
#else
            eval_params->clear();
            //eval_params->initRandom();
#endif
            eval_params->printHistgram();
            eval_params->writeFile();
            eval_params->writeFile("tmp.bin");
            std::cout << "0初期化したパラメータを出力" << std::endl;
        } else if (input == "cleanGame") {
            std::cout << "棋譜のあるフォルダへのパス : ";
            std::string file_path;
            std::cin >> file_path;
            cleanGames(file_path);
        } else if (input == "BonanzaMethod") {
            BonanzaMethodTrainer trainer("bonanza_method_settings.txt");
            trainer.train();
        } else if (input == "learnAsync") {
            RootstrapTrainer trainer("rootstrap_settings.txt");
            trainer.learnAsync();
        } else if (input == "learnSync") {
            RootstrapTrainer trainer("rootstrap_settings.txt");
            trainer.learnSync();
        } else if (input == "alphaZero") {
            AlphaZeroTrainer trainer("alphazero_settings.txt");
            trainer.learn();
#ifndef USE_MCTS
        } else if (input == "treeStrap") {
            TreestrapTrainer trainer("treestrap_settings.txt");
            trainer.startLearn();
#endif
        } else if (input == "printEvalParams") {
            eval_params->readFile();
            eval_params->printHistgram();
        } else if (input == "testMakeRandomPosition") {
            testMakeRandomPosition();
        } else if (input == "testKifuOutput") {
            testKifuOutput();
#ifndef USE_NN
        } else if (input == "convertAnotherEvalParams") {
            std::string path;
            std::cin >> path;
            std::string kkp_file_name;
            std::cin >> kkp_file_name;
            std::string kpp_file_name;
            std::cin >> kpp_file_name;
            readAnotherFormatFile(path, kkp_file_name, kpp_file_name);
            eval_params->writeFile();
#else
        } else if (input == "printBias") {
            eval_params->readFile("tmp.bin");
            eval_params->printBias();
#endif
        } else if (input == "testNN") {
            testNN();
        } else if (input == "testLearn") {
            AlphaZeroTrainer trainer("alphazero_settings.txt");
            trainer.testLearn();
        } else if (input == "testMirror") {
            testMirror();
        } else if (input == "testSFEN") {
            testSFENoutput();
        } else {
            std::cout << "Illegal input" << std::endl;
        }
    }
}

void USI::usi() {
#ifdef USE_NN
#ifdef USE_CATEGORICAL
    printf("id name kaitei_cat\n");
#else
    printf("id name kaitei_nn\n");
#endif
#else
    printf("id name Miacis\n");
#endif
    printf("id author Sakoda Shintaro\n");
	printf("option name byoyomi_margin type spin default 0 min 0 max 1000\n");
    usi_option.byoyomi_margin = 0;
	printf("option name random_turn type spin default 0 min 0 max 1000\n");
    usi_option.random_turn = 0;
    printf("option name thread_num type spin default 1 min 1 max %d\n", std::max(std::thread::hardware_concurrency(), 1U));
    usi_option.thread_num = 1;
    printf("option name draw_score type spin default -1 min -30000 max 30000\n");
    usi_option.draw_score = -1;
    printf("option name draw_turn type spin default 256 min 0 max 4096\n");
    usi_option.draw_turn = 256;

#ifdef USE_MCTS
    uint64_t d = (uint64_t)1e9;
    printf("option name playout_limit type spin default %llu min 1 max %llu\n", d, d);
    usi_option.playout_limit = d;
#else
    printf("option name depth_limit type spin default 64 min 1 max 64\n");
    usi_option.depth_limit = 64 * PLY;
#endif

    usi_option.USI_Hash = 256;
	printf("usiok\n");
}

void USI::isready() {
    eval_params->readFile();
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
            } else if (input == "draw_score") {
                std::cin >> input; //input == "value"となるはず
                std::cin >> usi_option.draw_score;
                return;
            } else if (input == "draw_turn") {
                std::cin >> input; //input == "value"となるはず
                std::cin >> usi_option.draw_turn;
                return;
#ifdef USE_MCTS
            } else if (input == "playout_limit") {
                std::cin >> input;
                std::cin >> usi_option.playout_limit;
                return;
#else
            } else if (input == "depth_limit") {
                std::cin >> input;
                std::cin >> usi_option.depth_limit;
                usi_option.depth_limit *= PLY;
                return;
#endif
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
    Searcher::stop_signal = false;
    Searcher::print_usi_info = true;
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
            Searcher::limit_msec = stoll(input);
        } else {
            int64_t binc, winc;
            std::cin >> input;
            binc = stoll(input);
            std::cin >> input; //input == "winc" となるはず
            std::cin >> input;
            winc = stoll(input);
            Searcher::limit_msec = binc;
        }
    } else if (input == "infinite") {
        //stop来るまで思考し続ける
        //思考時間をほぼ無限に
        Searcher::limit_msec = LLONG_MAX;
        
        //random_turnをなくす
        usi_option.random_turn = 0;
    } else if (input == "mate") {
        //詰み探索(未実装)
        std::cin >> input;
        if (input == "infinite") {
            //stop来るまで
            Searcher ::limit_msec = LLONG_MAX;
        } else {
            //思考時間が指定された場合
            //どう実装すればいいんだろう
        }
    }

    //思考開始
    //thinkを直接書くとstopコマンドを受け付けられなくなってしまうので
    //別スレッドに投げる
    thread_ = std::thread([&]() {
        Searcher searcher(usi_option.USI_Hash, usi_option.thread_num);
        auto result = searcher.think(root_);
        if (result.first == NULL_MOVE) {
            std::cout << "bestmove resign" << std::endl;
        } else {
            std::cout << "bestmove " << result.first << std::endl;
        }
    });
}

void USI::stop() {
    Searcher::stop_signal = true;
    while (!thread_.joinable());
    thread_.join();
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