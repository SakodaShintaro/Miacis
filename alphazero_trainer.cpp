#include"alphazero_trainer.hpp"
#include"position.hpp"
#include"MCTSearcher.hpp"
#include"operate_params.hpp"
#include"neural_network.hpp"
#include"parallel_MCTSearcher.hpp"
#include"game_generator.hpp"
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

const std::string LEARN_DIR = "learn_games";
const std::string TEST_DIR  = "test_games";
const std::string BEST_MODEL = "best.model";

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
        } else if (name == "momentum") {
            ifs >> MOMENTUM;
        } else if (name == "threshold(0.0~1.0)") {
            ifs >> THRESHOLD;
        } else if (name == "random_move_num") {
            ifs >> usi_option.random_turn;
        } else if (name == "draw_turn") {
            ifs >> usi_option.draw_turn;
        } else if (name == "draw_score") {
            ifs >> usi_option.draw_score;
        } else if (name == "lambda") {
            ifs >> replay_buffer_.lambda;
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
            ifs >> replay_buffer_.max_size;
        } else if (name == "first_wait") {
            ifs >> replay_buffer_.first_wait;
        } else if (name == "max_step_num") {
            ifs >> MAX_STEP_NUM;
        } else if (name == "playout_limit") {
            ifs >> usi_option.playout_limit;
        } else if (name == "parallel_num") {
            ifs >> PARALLEL_NUM;
        } else if (name == "validation_kifu_path") {
            ifs >> VALIDATION_KIFU_PATH;
        } else if (name == "validation_size") {
            ifs >> VALIDATION_SIZE;
        }
    }

    //その他オプションを学習用に設定
    usi_option.limit_msec = LLONG_MAX;
    usi_option.stop_signal = false;
    usi_option.byoyomi_margin = 0LL;

    //変数の初期化
    update_num_ = 0;

    //棋譜を保存するディレクトリの削除
    std::experimental::filesystem::remove_all("./learn_games");
    std::experimental::filesystem::remove_all("./test_games");

    //棋譜を保存するディレクトリの作成
#ifdef _MSC_VER
    _mkdir(LEARN_DIR);
    _mkdir(TEST_DIR);
#elif __GNUC__
    mkdir(LEARN_DIR.c_str(), ACCESSPERMS);
    mkdir(TEST_DIR.c_str(),  ACCESSPERMS);
#endif
}

void AlphaZeroTrainer::startLearn() {
    std::cout << "start alphaZero()" << std::endl;

    //局面インスタンスは一つ用意して都度局面を構成
    Position pos;

    //モデル読み込み
#ifdef USE_LIBTORCH
    torch::load(learning_model_, MODEL_PATH);
    torch::save(learning_model_, BEST_MODEL);
    torch::load(nn, MODEL_PATH);
#else
    learning_model_.load(MODEL_PATH);
    learning_model_.save(BEST_MODEL);
    nn->load(MODEL_PATH);
#endif

    //時間を初期化
    start_time_ = std::chrono::steady_clock::now();

    //変数の初期化
    update_num_ = 0;

    //ログファイルの設定
    std::ofstream log_file("alphazero_log.txt");
    log_file  << "time\tstep\tloss\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;
    std::cout << "time\tstep\tloss\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;

    //Optimizerの準備
#ifdef USE_LIBTORCH
    torch::optim::SGD optimizer(learning_model_->parameters(), LEARN_RATE);

    //自己対局をしてreplay_buffer_にデータを追加するインスタンス
    GameGenerator generator(0, PARALLEL_NUM, replay_buffer_, nn);
#else
    O::MomentumSGD optimizer(LEARN_RATE);
    optimizer.set_weight_decay(1e-4);
    optimizer.add(learning_model_);

    //自己対局をしてreplay_buffer_にデータを追加するインスタンス
    GameGenerator generator(0, PARALLEL_NUM, replay_buffer_, *nn);

    Graph g;
    Graph::set_default(g);
#endif
    std::thread gen_thread([&generator]() { generator.genGames(static_cast<int64_t>(1e10)); });

    for (int32_t step_num = 1; step_num <= MAX_STEP_NUM; step_num++) {
        //バッチサイズだけデータを選択
        std::vector<float> inputs;
        std::vector<PolicyTeacherType> policy_teachers;
        std::vector<ValueTeacherType> value_teachers;
        replay_buffer_.makeBatch(static_cast<int32_t>(BATCH_SIZE), inputs, policy_teachers, value_teachers);

        generator.gpu_mutex.lock();
#ifdef USE_LIBTORCH
        optimizer.zero_grad();
        auto loss = learning_model_->loss(inputs, policy_teachers, value_teachers);
        auto sum_loss = loss.first + loss.second;
        auto p_loss = loss.first.item<float>();
        auto v_loss = loss.second.item<float>();
        std::cout << elapsedTime() << "\t" << step_num << "\t" << sum_loss.item<float>() << "\t" << p_loss << "\t" << v_loss
                  << std::endl;
        log_file << elapsedHours() << "\t" << step_num << "\t" << sum_loss.item<float>() << "\t" << p_loss << "\t" << v_loss
                 << std::endl;
        sum_loss.backward();
        optimizer.step();
        torch::save(learning_model_, MODEL_PATH);
        torch::load(nn, MODEL_PATH);
        generator.gpu_mutex.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
#else
        g.clear();
        auto loss = learning_model_.loss(inputs, policy_teachers, value_teachers);
        optimizer.reset_gradients();
        loss.first.backward();
        loss.second.backward();
        optimizer.update();

        //学習情報の表示
        float p_loss = loss.first.to_float();
        float v_loss = loss.second.to_float();
        float sum_loss = POLICY_LOSS_COEFF * p_loss + VALUE_LOSS_COEFF * v_loss;
        std::cout << elapsedTime() << "\t" << step_num << "\t" << sum_loss << "\t" << p_loss << "\t" << v_loss << std::endl;
        log_file << elapsedHours() << "\t" << step_num << "\t" << sum_loss << "\t" << p_loss << "\t" << v_loss << std::endl;

        //書き出し
        learning_model_.save(MODEL_PATH);
        nn->load(MODEL_PATH);
#endif
        if (step_num % EVALUATION_INTERVAL == 0) {
            validation(step_num);
        }

        generator.gpu_mutex.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    usi_option.stop_signal = true;
    gen_thread.detach();

    std::cout << "finish alphaZero()" << std::endl;
}

void AlphaZeroTrainer::evaluate(int64_t step) {
    static std::ofstream eval_log;
    if (!eval_log) {
        eval_log.open("eval_log.txt");
        eval_log << "step\ttime\t勝率\t千日手\t超手数\t更新回数\t重複数\t次回のランダム数" << std::endl;
    }

    //設定を評価用に変える
    auto before_random_turn = usi_option.random_turn;
    usi_option.random_turn = EVALUATION_RANDOM_TURN;
    auto test_games = play(EVALUATION_GAME_NUM, false);

    //設定を戻す
    usi_option.random_turn = before_random_turn;

    //いくつか出力
    for (int32_t i = 0; i < std::min(4, (int32_t) test_games.size()); i++) {
        test_games[i].writeKifuFile(TEST_DIR);
    }

    double win_rate = 0.0;
    int32_t draw_repeat_num = 0, draw_over_limit_num = 0;
    for (int32_t i = 0; i < test_games.size(); i++) {
        if (test_games[i].result == Game::RESULT_DRAW_REPEAT) {
            draw_repeat_num++;
            test_games[i].result = (MAX_SCORE + MIN_SCORE) / 2;
        } else if (test_games[i].result == Game::RESULT_DRAW_OVER_LIMIT) {
            draw_over_limit_num++;
            test_games[i].result = (MAX_SCORE + MIN_SCORE) / 2;
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
#ifdef USE_LIBTORCH
        torch::save(learning_model_, BEST_MODEL);
#else
        learning_model_.save(BEST_MODEL);
#endif
        update_num_++;
    }
    eval_log << step << "\t" << elapsedHours() << "\t" << win_rate * 100.0 << "\t" << draw_repeat_num << "\t"
    << draw_over_limit_num << "\t" << update_num_ << "\t"
    << same_num << "\t" << (same_num == 0 ? --EVALUATION_RANDOM_TURN : ++EVALUATION_RANDOM_TURN);
}

std::vector<Game> AlphaZeroTrainer::play(int32_t game_num, bool eval) {
    std::vector<Game> games((unsigned long)game_num);

    //現在のパラメータ
#ifdef USE_LIBTORCH
    NeuralNetwork curr;
    torch::load(curr, MODEL_PATH);
    if (eval) {
        torch::load(nn, MODEL_PATH);
    }

    auto searcher1 = std::make_unique<ParallelMCTSearcher>(usi_option.USI_Hash, usi_option.thread_num, curr);
    auto searcher2 = (eval ? //searcher2は評価時にしか使わない
                      std::make_unique<ParallelMCTSearcher>(usi_option.USI_Hash, usi_option.thread_num, nn) :
                      nullptr);
#else
    NeuralNetwork<Tensor> curr;
    curr.load(MODEL_PATH);

    if (eval) {
        nn->load(BEST_MODEL);
    }

    auto searcher1 = std::make_unique<ParallelMCTSearcher>(usi_option.USI_Hash, usi_option.thread_num, curr);
    auto searcher2 = (eval ? //searcher2は評価時にしか使わない
            std::make_unique<ParallelMCTSearcher>(usi_option.USI_Hash, usi_option.thread_num, *nn) :
            nullptr);
#endif

    for (int32_t i = 0; i < game_num; i++) {
        Game& game = games[i];
        Position pos;

        while (true) {
            Move best_move;
            if (eval) {
                //評価時:iが偶数のときsearcher1が先手
                best_move = ((pos.turn_number() % 2) == (i % 2) ?
                                         searcher1->think(pos) :
                                         searcher2->think(pos));
            } else {
                //データ生成時:searcher1のみを使って自己対局
                best_move = searcher1->think(pos);
            }

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
            pos.print(false);
            game.moves.push_back(best_move);

            if (pos.turn_number() >= usi_option.draw_turn) { //長手数
                game.result = Game::RESULT_DRAW_OVER_LIMIT;
                break;
            }
        }
    }

    return games;
}

void AlphaZeroTrainer::validation(int64_t step_num) {
    static bool first = true;
    static std::vector<std::pair<std::string, TeacherType>> validation_data;
    static std::ofstream validation_log("alphazero_validation_log.txt");

    if (first) {
        //棋譜を読み込めるだけ読み込む
        auto games = loadGames(VALIDATION_KIFU_PATH, 100000);

        //データを局面単位にバラす
        std::vector<std::pair<std::string, TeacherType>> data_buffer;
        for (const auto& game : games) {
            Position pos;
            for (const auto& move : game.moves) {
                TeacherType teacher;
                teacher.policy = (uint32_t)move.toLabel();
#ifdef USE_CATEGORICAL
                assert(false);
#else
                teacher.value = (float)(pos.color() == BLACK ? game.result : MAX_SCORE + MIN_SCORE - game.result);
#endif
                data_buffer.emplace_back(pos.toSFEN(), teacher);
                pos.doMove(move);
            }
        }
        assert(data_buffer.size() >= VALIDATION_SIZE);

        //データをシャッフル
        std::default_random_engine engine(0);
        std::shuffle(data_buffer.begin(), data_buffer.end(), engine);

        //validationデータを確保
        for (int32_t i = 0; i < VALIDATION_SIZE; i++) {
            validation_data.push_back(data_buffer.back());
            data_buffer.pop_back();
        }

        //ラベルを記入
        validation_log << "step_num\telapsed_hours\tsum_loss\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;

        first = false;
    }
#ifdef USE_LIBTORCH
    torch::load(nn, MODEL_PATH);
#else
    nn->load(MODEL_PATH);
#endif

    int32_t num = 0;
    float policy_loss = 0.0, value_loss = 0.0;
    for (int32_t i = 0; (i + 1) * BATCH_SIZE <= validation_data.size(); i++, num++) {
        std::vector<float> inputs;
        std::vector<PolicyTeacherType> policy_teachers;
        std::vector<ValueTeacherType> value_teachers;

        Position pos;
        for (int32_t b = 0; b < BATCH_SIZE; b++) {
            const auto& datum = validation_data[i * BATCH_SIZE + b];
            pos.loadSFEN(datum.first);
            const auto& feature = pos.makeFeature();
            inputs.insert(inputs.end(), feature.begin(), feature.end());
            policy_teachers.push_back(datum.second.policy);
            value_teachers.push_back(datum.second.value);
        }

        auto loss = nn->loss(inputs, policy_teachers, value_teachers);
#ifdef USE_LIBTORCH
        policy_loss += loss.first.item<float>();
        value_loss  += loss.second.item<float>();
#else
        policy_loss += loss.first.to_float();
        value_loss  += loss.second.to_float();
#endif
    }
    policy_loss /= num;
    value_loss /= num;

    validation_log << step_num << "\t" << elapsedHours() << "\t" << POLICY_LOSS_COEFF * policy_loss + VALUE_LOSS_COEFF * value_loss
                   << "\t" << policy_loss << "\t" << value_loss << std::endl;
}