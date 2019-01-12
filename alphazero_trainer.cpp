#include"alphazero_trainer.hpp"
#include"position.hpp"
#include"MCTSearcher.hpp"
#include"operate_params.hpp"
#include"neural_network.hpp"
#include "alphazero_trainer.hpp"

#include<iomanip>
#include<algorithm>
#include<thread>
#include<mutex>
#include<climits>
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
        } else if (name == "learn_rate") {
            ifs >> LEARN_RATE;
        } else if (name == "learn_rate_decay") {
            ifs >> LEARN_RATE_DECAY;
        } else if (name == "momentum_decay") {
            ifs >> MOMENTUM_DECAY;
        } else if (name == "thread_num") {
            ifs >> THREAD_NUM;
            THREAD_NUM = std::min(std::max(2u, THREAD_NUM), std::thread::hardware_concurrency());
            usi_option.thread_num = THREAD_NUM;
        } else if (name == "threshold(0.0~1.0)") {
            ifs >> THRESHOLD;
        } else if (name == "random_move_num") {
            ifs >> usi_option.random_turn;
        } else if (name == "draw_turn") {
            ifs >> usi_option.draw_turn;
        } else if (name == "draw_score") {
            ifs >> usi_option.draw_score;
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
        } else if (name == "policy_loss_coeff") {
            ifs >> POLICY_LOSS_COEFF;
        } else if (name == "value_loss_coeff") {
            ifs >> VALUE_LOSS_COEFF;
        } else if (name == "max_stack_size") {
            ifs >> MAX_REPLAY_BUFFER_SIZE;
            replay_buffer_.reserve((unsigned long) MAX_REPLAY_BUFFER_SIZE);
        } else if (name == "max_step_num") {
            ifs >> MAX_STEP_NUM;
        } else if (name == "playout_limit") {
            ifs >> usi_option.playout_limit;
        }
    }

    //その他オプションを学習用に設定
    usi_option.limit_msec = LLONG_MAX;
    usi_option.stop_signal = false;
    usi_option.byoyomi_margin = 0LL;

    //変数の初期化
    update_num_ = 0;

    //評価関数読み込み
    learning_model_.load(MODEL_PATH);

//    //棋譜を保存するディレクトリの削除
//    std::experimental::filesystem::remove_all("./learn_games");
//    std::experimental::filesystem::remove_all("./test_games");
//
//    //棋譜を保存するディレクトリの作成
//#ifdef _MSC_VER
//    _mkdir("./learn_games");
//    _mkdir("./test_games");
//#elif __GNUC__
//    mkdir("./learn_games", ACCESSPERMS);
//    mkdir("./test_games", ACCESSPERMS);
//#endif
}

void AlphaZeroTrainer::startLearn() {
    std::cout << "start alphaZero()" << std::endl;

    //自己対局スレッドの作成
    std::vector<std::thread> slave_threads(THREAD_NUM - 1);
    for (uint32_t i = 0; i < THREAD_NUM - 1; i++) {
        slave_threads[i] = std::thread(&AlphaZeroTrainer::act, this);
    }

    //乱数の準備
    std::default_random_engine engine(0);

    //局面もインスタンスは一つ用意して都度局面を構成
    Position pos;

    auto start_learning_rate = LEARN_RATE;

    //初期化
    learning_model_.load(MODEL_PATH);
    nn->load(MODEL_PATH);

    //学習
    //時間を初期化
    start_time_ = std::chrono::steady_clock::now();

    MUTEX.lock();

    //変数の初期化
    update_num_ = 0;

    //学習率の初期化
    LEARN_RATE = start_learning_rate / BATCH_SIZE;

    //ログファイルの設定
    log_file_.open("alphazero_log.txt");
    print("経過時間");
    print("ステップ数");
    print("損失");
    print("Policy損失");
    print("Value損失");
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
    print(0.0);
    print(0.0);
    evaluate();
    std::cout << std::endl;
    log_file_ << std::endl;

    replay_buffer_.clear();
    replay_buffer_.reserve((unsigned long) MAX_REPLAY_BUFFER_SIZE);

    MUTEX.unlock();

    for (int32_t step_num = 1; step_num <= MAX_STEP_NUM; step_num++) {
        //ミニバッチ分勾配を貯める

        //損失初期化

        //バッチサイズだけデータを選択
        for (int32_t j = 0; j < BATCH_SIZE; j++) {
            if (replay_buffer_.size() <= BATCH_SIZE * 10) {
                j--;
                continue;
            }

            //ランダムに選ぶ
            MUTEX.lock();
            //サイズがオーバーしていたら減らす
            if ((int64_t) replay_buffer_.size() >= MAX_REPLAY_BUFFER_SIZE) {
                auto diff = replay_buffer_.size() - MAX_REPLAY_BUFFER_SIZE;
                replay_buffer_.erase(replay_buffer_.begin(),
                                     replay_buffer_.begin() + diff + MAX_REPLAY_BUFFER_SIZE / 10);
                replay_buffer_.shrink_to_fit();
            }

            auto random = engine() % replay_buffer_.size();
            auto datum = replay_buffer_[random];
            MUTEX.unlock();

            //局面を復元
            pos.loadSFEN(datum.first);
        }

        MUTEX.lock();
        //学習
        LossType loss{};

        //学習情報の表示
        timestamp();
        print(step_num);
        print(POLICY_LOSS_COEFF * loss[0] + VALUE_LOSS_COEFF * loss[1]);
        print(loss[0]);
        print(loss[1]);

        //評価と書き出し
        if (step_num % EVALUATION_INTERVAL == 0 || step_num == MAX_STEP_NUM) {
            evaluate();
            learning_model_.save("tmp" + std::to_string(step_num) + ".bin");
        }

        std::cout << std::endl;
        log_file_ << std::endl;

        //学習率の減衰
        LEARN_RATE *= LEARN_RATE_DECAY;

        MUTEX.unlock();
    }

    log_file_.close();

    usi_option.stop_signal = true;
    for (uint32_t i = 0; i < THREAD_NUM - 1; i++) {
        slave_threads[i].join();
        printf("%2dスレッドをjoin\n", i);
    }

    std::cout << "finish alphaZero()" << std::endl;
}

void AlphaZeroTrainer::testLearn() {
    std::cout << "start testLearn()" << std::endl;

    //パラメータ初期化
    learning_model_.init();

    //自己対局
    auto games = parallelPlay(1);

    std::cout << std::fixed;

    for (auto &game : games) {
        if (game.result == Game::RESULT_DRAW_REPEAT || game.result == Game::RESULT_DRAW_OVER_LIMIT) {
            if (!USE_DRAW_GAME) { //使わないと指定されていたらスキップ
                continue;
            } else { //そうでなかったら結果を後手勝ちにして
                game.result = Game::RESULT_WHITE_WIN;
            }
        }

        BATCH_SIZE = game.moves.size();

        pushOneGame(game);
    }

    //局面もインスタンスは一つ用意して都度局面を構成
    Position pos;

    //学習
    O::SGD optimizer(0.01);
    optimizer.add(learning_model_);

    Graph g;
    Graph::set_default(g);

    for (int32_t step_num = 1; step_num <= MAX_STEP_NUM; step_num++) {
        //ミニバッチ分勾配を貯める
        std::vector<float> inputs, value_teachers;
        std::vector<uint32_t> policy_teachers;
        for (const auto& datum : replay_buffer_) {
            auto sfen = datum.first;
            auto teacher = datum.second;
            pos.loadSFEN(sfen);

            for (const auto& e : pos.makeFeature()) {
                inputs.push_back(e);
            }
            policy_teachers.push_back(teacher.first);
            value_teachers.push_back(teacher.second);
        }

        g.clear();
        auto losses = learning_model_.loss(inputs, policy_teachers, value_teachers, BATCH_SIZE);
        std::cout << losses.first.to_float() << " " << losses.second.to_float() << std::endl;
        optimizer.reset_gradients();
        losses.first.backward();
        losses.second.backward();
        optimizer.update();
    }

    learning_model_.save(MODEL_PATH);
    nn->load(MODEL_PATH);

    for (const auto &game : games) {
        Position position;
        for (int32_t i = 0; i < game.moves.size(); i++) {
            position.print();
            position.doMove(game.moves[i]);
        }
    }

    std::cout << "finish testLearn()" << std::endl;
}

void AlphaZeroTrainer::act() {
    //探索クラスを生成
    NeuralNetwork<Tensor> model;
    model.init();
    model.load(MODEL_PATH);
    auto searcher = std::make_unique<MCTSearcher<Tensor>>(usi_option.USI_Hash, 1, model);

    //停止信号が来るまでループ
    while (!usi_option.stop_signal) {
        //棋譜を生成
        Game game;

        Position pos;

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

        pushOneGame(game);
    }
}

void AlphaZeroTrainer::evaluate() {
    //対局するパラメータを準備
    nn->load(MODEL_PATH);

    //設定を評価用に変える
    auto before_random_turn = usi_option.random_turn;
    usi_option.random_turn = EVALUATION_RANDOM_TURN;
    usi_option.train_mode = false;
    auto test_games = parallelPlay(EVALUATION_GAME_NUM);

    //設定を戻す
    usi_option.random_turn = before_random_turn;
    usi_option.train_mode = true;

    //いくつか出力
    for (int32_t i = 0; i < std::min(4, (int32_t) test_games.size()); i++) {
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
        learning_model_.save(MODEL_PATH);
        update_num_++;
    }
    print(win_rate * 100.0);
    print(draw_repeat_num);
    print(draw_over_limit_num);
    print(update_num_);
    print(same_num);
    print(same_num == 0 ? --EVALUATION_RANDOM_TURN : ++EVALUATION_RANDOM_TURN);
}

void AlphaZeroTrainer::pushOneGame(Game &game) {
    Position pos;

    //まずは最終局面まで動かす
    for (auto move : game.moves) {
        pos.doMove(move);
    }

    assert(Game::RESULT_WHITE_WIN <= game.result && game.result <= Game::RESULT_BLACK_WIN);

    //先手から見た勝率,分布.指数移動平均で動かしていく.最初は結果によって初期化(0 or 0.5 or 1)
    double win_rate_for_black = game.result;

    for (int32_t i = (int32_t) game.moves.size() - 1; i >= 0; i--) {
        //i番目の指し手を教師とするのは1手戻した局面
        pos.undo();

        //探索結果を先手から見た値に変換
        double curr_win_rate = (pos.color() == BLACK ? game.moves[i].score : 1.0 - game.moves[i].score);

        //混合
        win_rate_for_black = LAMBDA * win_rate_for_black + (1.0 - LAMBDA) * curr_win_rate;

#ifdef USE_CATEGORICAL
        //手番から見た分布を得る
        auto teacher_dist = onehotDist(pos.color() == BLACK ? win_rate_for_black : 1.0 - win_rate_for_black);

        //teacherにコピーする
        for (int32_t j = 0; j < BIN_SIZE; j++) {
            game.teachers[i][POLICY_DIM + j] = teacher_dist[j];
        }
#else
        //teacherにコピーする
        game.teachers[i].second = (CalcType) (pos.color() == BLACK ? win_rate_for_black : 1.0 - win_rate_for_black);
#endif

        //スタックに詰める
        MUTEX.lock();
        replay_buffer_.emplace_back(pos.toSFEN(), game.teachers[i]);
        MUTEX.unlock();
    }
}

std::vector<Game> AlphaZeroTrainer::parallelPlay(int32_t game_num) {
    std::vector<Game> games((unsigned long)game_num);
    std::atomic<int32_t> index;
    index = 0;

    std::vector<std::thread> threads;
    for (int32_t i = 0; i < (int32_t)usi_option.thread_num; i++) {
        threads.emplace_back([&]() {
            auto searcher1 = std::make_unique<MCTSearcher<Node>>(usi_option.USI_Hash, 1, learning_model_);
            auto searcher2 = std::make_unique<MCTSearcher<Tensor>>(usi_option.USI_Hash, 1, *nn);
            while (true) {
                int32_t curr_index = index++;
                if (curr_index >= game_num) {
                    return;
                }
                Game& game = games[curr_index];
                game.moves.reserve((unsigned long)usi_option.draw_turn);
                game.teachers.reserve((unsigned long)usi_option.draw_turn);
                Position pos;

                while (true) {
                    //iが偶数のときpos_cが先手
                    auto move_and_teacher = ((pos.turn_number() % 2) == (curr_index % 2) ?
                                             searcher1->think(pos) :
                                             searcher2->think(pos));
                    Move best_move = move_and_teacher.first;
                    TeacherType teacher = move_and_teacher.second;

                    if (best_move == NULL_MOVE) { //NULL_MOVEは投了を示す
                        game.result = (pos.color() == BLACK ? Game::RESULT_WHITE_WIN : Game::RESULT_BLACK_WIN);
                        break;
                    }

                    if (!pos.isLegalMove(best_move)) {
                        pos.printForDebug();
                        best_move.printWithScore();
                        assert(false);
                    }
                    pos.doMove(best_move);
                    pos.print();
                    game.moves.push_back(best_move);
                    game.teachers.push_back(teacher);

                    if (pos.turn_number() >= usi_option.draw_turn) { //長手数
                        game.result = Game::RESULT_DRAW_OVER_LIMIT;
                        break;
                    }
                }
            }
        });
    }
    for (int32_t i = 0; i < (int32_t)usi_option.thread_num; i++) {
        threads[i].join();
    }
    return games;
}
