#include"alphazero_trainer.hpp"
#include"rootstrap_trainer.hpp"
#include"position.hpp"
#include"searcher.hpp"
#include"eval_params.hpp"
#include"operate_params.hpp"
#include<iomanip>
#include<algorithm>
#include<thread>
#include<experimental/filesystem>
#ifdef _MSC_VER
#include<direct.h>
#elif __GNUC__
#include<sys/stat.h>
#endif

static std::mutex MUTEX;

AlphaZeroTrainer::AlphaZeroTrainer(std::string settings_file_path) {
    //オプションをファイルから読み込む
    std::ifstream ifs(settings_file_path);
    if (!ifs) {
        std::cerr << "fail to open setting_file(" << settings_file_path << ")" << std::endl;
        assert(false);
    }

    std::string name;
    while (ifs >> name) {
        if (name == "batch_size") {
            ifs >> BATCH_SIZE;
        } else if (name == "optimizer") {
            ifs >> OPTIMIZER_NAME;
            if (!isLegalOptimizer()) {
                std::cerr << "optimizerが不正" << std::endl;
                assert(false);
            }
        } else if (name == "learn_rate") {
            ifs >> LEARN_RATE;
        } else if (name == "learn_rate_decay") {
            ifs >> LEARN_RATE_DECAY;
        } else if (name == "momentum_decay") {
            ifs >> MOMENTUM_DECAY;
        } else if (name == "thread_num") {
            ifs >> THREAD_NUM;
            THREAD_NUM = std::min(std::max(1u, THREAD_NUM), std::thread::hardware_concurrency());
            usi_option.thread_num = THREAD_NUM;
        } else if (name == "threshold(0.0~1.0)") {
            ifs >> THRESHOLD;
        } else if (name == "random_move_num") {
            ifs >> usi_option.random_turn;
        } else if (name == "draw_turn") {
            ifs >> usi_option.draw_turn;
        } else if (name == "draw_score") {
            ifs >> usi_option.draw_score;
        } else if (name == "learn_mode(0or1)") {
            ifs >> LEARN_MODE;
        } else if (name == "deep_coefficient") {
            ifs >> DEEP_COEFFICIENT;
        } else if (name == "lambda") {
            ifs >> LAMBDA;
        } else if (name == "use_draw_game") {
            ifs >> USE_DRAW_GAME;
        } else if (name == "USI_Hash") {
            ifs >> usi_option.USI_Hash;
        } else if (name == "evaluation_game_num") {
            ifs >> EVALUATION_GAME_NUM;
        } else if (name == "evaluation_interval") {
            ifs >> EVALUATION_INTERVAL;
        } else if (name == "evaluation_random_turn") {
            ifs >> EVALUATION_RANDOM_TURN;
#ifdef USE_NN
        } else if (name == "policy_loss_coeff") {
            ifs >> POLICY_LOSS_COEFF;
        } else if (name == "value_loss_coeff") {
            ifs >> VALUE_LOSS_COEFF;
#endif
        } else if (name == "max_stack_size") {
            ifs >> MAX_STACK_SIZE;
            position_pool_.reserve((unsigned long)MAX_STACK_SIZE);
        } else if (name == "max_step_num") {
            ifs >> MAX_STEP_NUM;
#ifdef USE_MCTS
        } else if (name == "playout_limit") {
            ifs >> usi_option.playout_limit;
#else
        } else if (name == "search_depth") {
            ifs >> usi_option.depth_limit;
            usi_option.depth_limit *= PLY;
#endif
        }
    }

    //その他オプションを学習用に設定
    Searcher::limit_msec = LLONG_MAX;
    Searcher::stop_signal = false;
    usi_option.byoyomi_margin = 0LL;

    //変数の初期化
    update_num_ = 0;

    //評価関数読み込み
    eval_params->readFile("tmp.bin");
#ifndef USE_NN
    learning_parameters = std::make_unique<EvalParams<LearnEvalType>>();
    learning_parameters->copy(*eval_params);
#endif

    //Optimizerに合わせて必要なものを準備
    if (OPTIMIZER_NAME == "MOMENTUM") {
        pre_update_ = std::make_unique<EvalParams<LearnEvalType>>();
    }

    //棋譜を保存するディレクトリの削除
    std::experimental::filesystem::remove_all("./learn_games");
    std::experimental::filesystem::remove_all("./test_games");

    //棋譜を保存するディレクトリの作成
#ifdef _MSC_VER
    _mkdir("./learn_games");
    _mkdir("./test_games");
#elif __GNUC__
    mkdir("./learn_games", ACCESSPERMS);
    mkdir("./test_games", ACCESSPERMS);
#endif
}

void AlphaZeroTrainer::learn() {
    std::cout << "start alphaZero()" << std::endl;

    //自己対局スレッドの作成
    std::vector<std::thread> slave_threads(THREAD_NUM - 1);
    for (uint32_t i = 0; i < THREAD_NUM - 1; i++) {
        slave_threads[i] = std::thread(&AlphaZeroTrainer::learnSlave, this);
    }

    //乱数の準備
    std::random_device seed;
    std::default_random_engine engine(seed());

    //局面もインスタンスは一つ用意して都度局面を構成
    Position pos(*eval_params);

    auto start_learning_rate = LEARN_RATE;

    //初期の対戦相手を固定する
    eval_params->initRandom();
    eval_params->writeFile("first_target.bin");

    //学習
    for (int32_t i = 1; i <= 100; i++) {
        //時間を初期化
        start_time_ = std::chrono::steady_clock::now();

        MUTEX.lock();

        //初期の対戦相手をmodel.binにコピー
        eval_params->readFile("first_target.bin");
        eval_params->writeFile();

        //パラメータの初期化
        eval_params->initRandom();
        eval_params->writeFile("before_learn" + std::to_string(i) + ".bin");
#ifndef USE_NN
        learning_parameters->copy(*eval_params);
#endif

        //変数の初期化
        update_num_ = 0;

        //学習率の初期化
        LEARN_RATE = start_learning_rate / BATCH_SIZE;

        //ログファイルの設定
        log_file_.open("alphazero_log" + std::to_string(i) + ".txt");
        print("経過時間");
        print("ステップ数");
        print("損失");
#ifdef USE_NN
        print("Policy損失");
        print("Value損失");
#endif
        print("最大更新量");
        print("総和更新量");
        print("最大パラメータ");
        print("総和パラメータ");
        print("勝率");
        print("千日手");
        print("512手");
        print("勝ち越し数");
        print("重複数");
        print("次のrandom_turn");
        log_file_ << std::endl << std::fixed;
        std::cout << std::endl << std::fixed;

        //0回目を入れてみる
        timestamp();
        print(0);
        print(0.0);
#ifdef USE_NN
        print(0.0);
        print(0.0);
#endif
        print(0.0);
        print(0.0);
        print(eval_params->maxAbs());
        print(eval_params->sumAbs());
        evaluate();
        std::cout << std::endl;
        log_file_ << std::endl;

        position_pool_.clear();
        position_pool_.reserve((unsigned long)MAX_STACK_SIZE);

        MUTEX.unlock();

        for (int32_t step_num = 1; step_num <= MAX_STEP_NUM; step_num++) {
            //ミニバッチ分勾配を貯める
            auto grad = std::make_unique<EvalParams<LearnEvalType>>();
#ifdef USE_NN
            std::array<double, 2> loss{ 0.0, 0.0 };
#else
            double loss = 0.0;
#endif
            for (int32_t j = 0; j < BATCH_SIZE; j++) {
                if (position_pool_.size() <= BATCH_SIZE * 10) {
                    j--;
                    continue;
                }

                //ランダムに選ぶ
                MUTEX.lock();
                //サイズがオーバーしていたら減らす
                if ((int64_t)position_pool_.size() >= MAX_STACK_SIZE) {
                    auto diff = position_pool_.size() - MAX_STACK_SIZE;
                    position_pool_.erase(position_pool_.begin(), position_pool_.begin() + diff + MAX_STACK_SIZE / 10);
                    position_pool_.shrink_to_fit();
                }

                auto random = engine() % position_pool_.size();
                auto data = position_pool_[random];
                MUTEX.unlock();

                //局面を復元
                pos.loadSFEN(data.first);

                //勾配を計算
                loss += addGrad(*grad, pos, data.second);
            }
            loss /= BATCH_SIZE;

            MUTEX.lock();
            //学習
#ifdef USE_NN
            updateParams(*eval_params, *grad);
#else
            updateParams(*learning_parameters, *grad);
            eval_params->copy(*learning_parameters);
#endif
            //学習情報の表示
            timestamp();
            print(step_num);
#ifdef USE_NN
            print(POLICY_LOSS_COEFF * loss[0] + VALUE_LOSS_COEFF * loss[1]);
            print(loss[0]);
            print(loss[1]);
#else
            print(loss);
#endif
            if (step_num < EVALUATION_INTERVAL) {
                print(LEARN_RATE * grad->maxAbs());
                print(LEARN_RATE * grad->sumAbs());
                print(eval_params->maxAbs());
                print(eval_params->sumAbs());
            }

            //評価と書き出し
            if (step_num % EVALUATION_INTERVAL == 0 || step_num == MAX_STEP_NUM) {
                print(LEARN_RATE * grad->maxAbs());
                print(LEARN_RATE * grad->sumAbs());
                print(eval_params->maxAbs());
                print(eval_params->sumAbs());

                evaluate();
                eval_params->writeFile("tmp" + std::to_string(i) + "_" + std::to_string(step_num) + ".bin");
            }

            std::cout << std::endl;
            log_file_ << std::endl;

            //学習率の減衰
            LEARN_RATE *= LEARN_RATE_DECAY;

            MUTEX.unlock();
        }

        eval_params->writeFile("tmp" + std::to_string(i) + ".bin");
        log_file_.close();
    }

    Searcher::stop_signal = true;
    for (uint32_t i = 0; i < THREAD_NUM - 1; i++) {
        slave_threads[i].join();
        printf("%2dスレッドをjoin\n", i);
    }

    std::cout << "finish alphaZero()" << std::endl;
}

void AlphaZeroTrainer::testLearn() {
    std::cout << "start testLearn()" << std::endl;

    //パラメータ初期化
    eval_params->initRandom();
#ifndef USE_NN
    learning_parameters->copy(*eval_params);
#endif

    //自己対局
    auto games = RootstrapTrainer::parallelPlay(*eval_params, *eval_params, 1);

    std::cout << std::fixed;

    for (auto& game : games) {
        if (game.result == Game::RESULT_DRAW_REPEAT || game.result == Game::RESULT_DRAW_OVER_LIMIT) {
            if (!USE_DRAW_GAME) { //使わないと指定されていたらスキップ
                continue;
            } else { //そうでなかったら結果を0.5にして使用
                game.result = 0.5;
            }
        }

        game.moves.resize((unsigned long)(usi_option.random_turn + 1));

        if (LEARN_MODE == ELMO_LEARN) {
            pushOneGame(game);
        } else if (LEARN_MODE == TD_LEAF_LAMBDA) {
            pushOneGameReverse(game);
        } else { //ここには来ないはず
            assert(false);
        }

        Position pos(*eval_params);
        for (int32_t i = 0; i < game.moves.size(); i++) {
            if (pos.turn_number() >= usi_option.random_turn) {
                pos.print();
#ifdef USE_NN
#else
                std::cout << (pos.color() == BLACK ? game.teachers[i] : 1.0 - game.teachers[i]) << std::endl;
#endif
            }
            pos.doMove(game.moves[i]);
        }
    }

    //学習率の初期化
    LEARN_RATE /= position_pool_.size();
    
    //局面もインスタンスは一つ用意して都度局面を構成
    Position pos(*eval_params);

    //初期の対戦相手を固定する
    eval_params->writeFile("first_target.bin");

    //学習
    for (int32_t step_num = 1; step_num <= MAX_STEP_NUM; step_num++) {
        //ミニバッチ分勾配を貯める
        auto grad = std::make_unique<EvalParams<LearnEvalType>>();
        LossType loss = {};
        for (const auto& data : position_pool_) {
            pos.loadSFEN(data.first);

            loss += addGrad(*grad, pos, data.second);
        }
        loss /= position_pool_.size();

        //学習
#ifdef USE_NN
        updateParams(*eval_params, *grad);
#else
        updateParams(*learning_parameters, *grad);
        eval_params->copy(*learning_parameters);
#endif
        //学習情報の表示
#ifdef USE_NN
        std::cout << std::setw(4) << step_num << " " << loss[0] << " " << loss[1] << std::endl;
#else
        std::cout << std::setw(4) << step_num << " " << loss << std::endl;
#endif
    }

    for (const auto& game : games) {
        Position position(*eval_params);
        for (int32_t i = 0; i < game.moves.size(); i++) {
            if (position.turn_number() >= usi_option.random_turn) {
                position.print();
#ifndef USE_NN
                std::cout << (position.color() == BLACK ? game.teachers[i] : 1.0 - game.teachers[i]) << std::endl;
#endif
            }
            position.doMove(game.moves[i]);
        }
    }

    std::cout << "finish testLearn()" << std::endl;
}

void AlphaZeroTrainer::learnSlave() {
    //探索クラスを生成
    auto searcher = std::make_unique<Searcher>(usi_option.USI_Hash, 1);

    //停止信号が来るまでループ
    while (!Searcher::stop_signal) {
        //棋譜を生成
        Game game;

        Position pos(*eval_params);

        while (true) {
            auto search_result = searcher->think(pos);
            Move best_move = search_result.first;
            TeacherType teacher = search_result.second;

            if (best_move == NULL_MOVE) { //NULL_MOVEは投了を示す
                game.result = (pos.color() == BLACK ? Game::RESULT_WHITE_WIN : Game::RESULT_BLACK_WIN);
                break;
            }

            pos.doMove(best_move);
            game.moves.push_back(best_move);
            game.teachers.push_back(teacher);

#ifndef USE_NN
            Score repeat_score;
            if (pos.isRepeating(repeat_score)) { //繰り返し
                if (isMatedScore(repeat_score)) {
                    if (repeat_score == MAX_SCORE + 1) {
                        std::cout << "連続王手の千日手" << std::endl;
                        pos.printForDebug();
                        std::this_thread::sleep_for(std::chrono::seconds(6000000));
                        assert(false);
                    }
                } else {
                    game.result = Game::RESULT_DRAW_REPEAT;
                    break;
                }
            }
#endif

            if (pos.turn_number() >= usi_option.draw_turn) { //長手数
                game.result = Game::RESULT_DRAW_OVER_LIMIT;
                break;
            }
        }

        //生成した棋譜を学習用データに加工してstackへ詰め込む
        //引き分けを処理する
        if (game.result == Game::RESULT_DRAW_REPEAT || game.result == Game::RESULT_DRAW_OVER_LIMIT) {
            if (!USE_DRAW_GAME) { //使わないと指定されていたらスキップ
                continue;
            } else { //そうでなかったら結果を0.5にして使用
                game.result = 0.5;
            }
        }

        if (LEARN_MODE == ELMO_LEARN) {
            pushOneGame(game);
        } else if (LEARN_MODE == TD_LEAF_LAMBDA) {
            pushOneGameReverse(game);
        } else { //ここには来ないはず
            assert(false);
        }
    }
}

void AlphaZeroTrainer::evaluate() {
    //対局するパラメータを準備
    auto opponent_parameters_ = std::make_unique<EvalParams<DefaultEvalType>>();
    opponent_parameters_->readFile();

    //設定を評価用に変える
    auto before_random_turn = usi_option.random_turn;
    usi_option.random_turn = EVALUATION_RANDOM_TURN;
    Searcher::train_mode = false;
    auto test_games = RootstrapTrainer::parallelPlay(*eval_params, *opponent_parameters_, EVALUATION_GAME_NUM);

    //設定を戻す
    usi_option.random_turn = before_random_turn;
    Searcher::train_mode = true;

    //いくつか出力
    for (int32_t i = 0; i < std::min(4, (int32_t)test_games.size()); i++) {
        test_games[i].writeKifuFile("./test_games/");
    }

    double win_rate = 0.0;
    int32_t draw_repeat_num = 0, draw_over_limit_num = 0;
    for (int32_t i = 0; i < test_games.size(); i++) {
        if (test_games[i].result == Game::RESULT_DRAW_REPEAT) {
            draw_repeat_num++;
            test_games[i].result = 0.5;
        } else if (test_games[i].result == Game::RESULT_DRAW_OVER_LIMIT) {
            draw_over_limit_num++;
            test_games[i].result = 0.5;
        }
        win_rate += (i % 2 == 0 ? test_games[i].result : 1.0 - test_games[i].result);
    }

    //重複の確認をしてみる
    int32_t same_num = 0;
    for (int32_t i = 0; i < test_games.size(); i++) {
        for (int32_t j = i + 1; j < test_games.size(); j++) {
            if (test_games[i].moves.size() != test_games[i].moves.size()) {
                continue;
            }
            bool same = true;
            for (int32_t k = 0; k < test_games[i].moves.size(); k++) {
                if (test_games[i].moves[k] != test_games[j].moves[k]) {
                    same = false;
                    break;
                }
            }
            if (same) {
                same_num++;
            }
        }
    }
    win_rate /= test_games.size();

    if (win_rate >= THRESHOLD) {
        eval_params->writeFile();
        update_num_++;
    }
    print(win_rate * 100.0);
    print(draw_repeat_num);
    print(draw_over_limit_num);
    print(update_num_);
    print(same_num);
    print(same_num == 0 ? --EVALUATION_RANDOM_TURN : ++EVALUATION_RANDOM_TURN);
}

void AlphaZeroTrainer::pushOneGame(Game& game) {
    Position pos(*eval_params);

    for (int32_t i = 0; i < game.moves.size(); i++) {
        const Move& move = game.moves[i];
        if (move.score == MIN_SCORE) {
            pos.doMove(move);
            continue;
        }

        //教師信号を計算
        double result_for_turn = (pos.color() == BLACK ? game.result : 1.0 - game.result);

#ifdef USE_NN
        double teacher_signal = DEEP_COEFFICIENT * game.moves[i].score + (1 - DEEP_COEFFICIENT) * result_for_turn;

#ifdef USE_CATEGORICAL
        auto teacher_dist = onehotDist(teacher_signal);
        std::copy(teacher_dist.begin(), teacher_dist.end(), &game.teachers[i][POLICY_DIM]);
#else
        game.teachers[i][POLICY_DIM] = (CalcType)teacher_signal;
#endif
#else
        game.teachers[i] = (DEEP_COEFFICIENT * game.teachers[i] + (1 - DEEP_COEFFICIENT) * result_for_turn);
#endif
        //pos.print();
        //std::cout << pos.toSFEN() << ", " << game.teachers[i] << std::endl;

        //スタックに詰める
        MUTEX.lock();
        position_pool_.emplace_back(pos.toSFEN(), game.teachers[i]);
        MUTEX.unlock();

        assert(pos.isLegalMove(move));
        pos.doMove(move);
    }
}

void AlphaZeroTrainer::pushOneGameReverse(Game& game) {
    Position pos(*eval_params);

    //まずは最終局面まで動かす
    for (auto move : game.moves) {
        pos.doMove(move);
    }

    assert(Game::RESULT_WHITE_WIN <= game.result && game.result <= Game::RESULT_BLACK_WIN);

    //先手から見た勝率,分布.指数移動平均で動かしていく.最初は結果によって初期化(0 or 0.5 or 1)
    double win_rate_for_black = game.result;

    for (int32_t i = (int32_t)game.moves.size() - 1; i >= 0; i--) {
        //i番目の指し手が対応するのは1手戻した局面
        pos.undo();

        if (game.moves[i].score == MIN_SCORE) {
            //ランダムムーブなので終了
            break;
        }

        //探索結果を先手から見た値に変換
#ifdef USE_NN
        double curr_win_rate = (pos.color() == BLACK ? game.moves[i].score : 1.0 - game.moves[i].score);
#else
        double curr_win_rate = (pos.color() == BLACK ? game.teachers[i] : 1.0 - game.teachers[i]);
#endif

        //混合
        win_rate_for_black = LAMBDA * win_rate_for_black + (1.0 - LAMBDA) * curr_win_rate;

#ifdef USE_NN
#ifdef USE_CATEGORICAL
        //手番から見た分布を得る
        auto teacher_dist = onehotDist(pos.color() == BLACK ? win_rate_for_black : 1.0 - win_rate_for_black);

        //teacherにコピーする
        for (int32_t j = 0; j < BIN_SIZE; j++) {
            game.teachers[i][POLICY_DIM + j] = teacher_dist[j];
        }
#else
        //teacherにコピーする
        game.teachers[i][POLICY_DIM] = (CalcType)(pos.color() == BLACK ? win_rate_for_black : 1.0 - win_rate_for_black);
#endif
#else
        game.teachers[i] = (pos.color() == BLACK ? win_rate_for_black : 1.0 - win_rate_for_black);
#endif

        //スタックに詰める
        MUTEX.lock();
        position_pool_.emplace_back(pos.toSFEN(), game.teachers[i]);
        MUTEX.unlock();
    }
}