#include"learn.hpp"
#include"replay_buffer.hpp"
#include"usi_options.hpp"
#include"game_generator.hpp"
#include"operate_params.hpp"
#include<thread>
#include<climits>

void alphaZero() {
    auto start_time_ = std::chrono::steady_clock::now();

    //オプションをファイルから読み込む
    std::ifstream ifs("alphazero_settings.txt");
    if (!ifs) {
        std::cerr << "fail to open alphazero_settings.txt" << std::endl;
        assert(false);
    }

    float LEARN_RATE = -1;
    double LEARN_RATE_DECAY = -1;
    double MOMENTUM = -1;
    int64_t BATCH_SIZE = -1;
    float POLICY_LOSS_COEFF = -1;
    float VALUE_LOSS_COEFF = -1;
    bool USE_DRAW_GAME = false;
    int64_t EVALUATION_INTERVAL = -1;
    int64_t MAX_STEP_NUM = -1;
    int64_t PARALLEL_NUM = -1;
    std::string VALIDATION_KIFU_PATH;
    int64_t VALIDATION_SIZE = -1;

    //学習中のモデル
#ifdef USE_LIBTORCH
    NeuralNetwork learning_model_;
#else
    NeuralNetwork<Node> learning_model_;
#endif

    //リプレイバッファ
    ReplayBuffer replay_buffer_;

    //ファイルを読み込んで値を設定
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
        } else if (name == "random_move_num") {
            ifs >> usi_option.random_turn;
        } else if (name == "draw_turn") {
            ifs >> usi_option.draw_turn;
        } else if (name == "draw_score") {
            ifs >> usi_option.draw_score;
        } else if (name == "use_draw_game") {
            ifs >> USE_DRAW_GAME;
        } else if (name == "USI_Hash") {
            ifs >> usi_option.USI_Hash;
        } else if (name == "policy_loss_coeff") {
            ifs >> POLICY_LOSS_COEFF;
        } else if (name == "value_loss_coeff") {
            ifs >> VALUE_LOSS_COEFF;
        } else if (name == "lambda") {
            ifs >> replay_buffer_.lambda;
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
        } else if (name == "evaluation_interval") {
            ifs >> EVALUATION_INTERVAL;
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

    //validation_dataの準備
    std::ofstream validation_log("alphazero_validation_log.txt");
    validation_log << "step_num\telapsed_hours\tsum_loss\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;

    //棋譜を読み込めるだけ読み込む
    auto games = loadGames(VALIDATION_KIFU_PATH, 100000);

    //データを局面単位にバラす
    std::vector<std::pair<std::string, TeacherType>> validation_data;
    for (const auto& game : games) {
        Position pos;
        for (const auto& move : game.moves) {
            TeacherType teacher;
            teacher.policy = (uint32_t) move.toLabel();
#ifdef USE_CATEGORICAL
            assert(false);
            teacher.value = valueToIndex((pos.color() == BLACK ? game.result : MAX_SCORE + MIN_SCORE - game.result));
#else
            teacher.value = (float) (pos.color() == BLACK ? game.result : MAX_SCORE + MIN_SCORE - game.result);
#endif
            validation_data.emplace_back(pos.toSFEN(), teacher);
            pos.doMove(move);
        }
    }
    assert(validation_data.size() >= VALIDATION_SIZE);

    //データをシャッフルして必要量以外を削除
    std::default_random_engine engine(0);
    std::shuffle(validation_data.begin(), validation_data.end(), engine);
    validation_data.erase(validation_data.begin() + VALIDATION_SIZE);

    //局面インスタンスは一つ用意して都度局面を構成
    Position pos;

    //モデル読み込み
#ifdef USE_LIBTORCH
    torch::load(learning_model_, MODEL_PATH);
    torch::save(learning_model_, MODEL_PREFIX + "_best.model");
    torch::load(nn, MODEL_PATH);
#else
    learning_model_.load(MODEL_PATH);
    learning_model_.save(MODEL_PREFIX + "_best.model");
    nn->load(MODEL_PATH);
#endif

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
        std::cout << elapsedTime(start_time_) << "\t" << step_num << "\t" << sum_loss.item<float>() << "\t" << p_loss
                  << "\t" << v_loss << std::endl;
        log_file << elapsedHours(start_time_) << "\t" << step_num << "\t" << sum_loss.item<float>() << "\t" << p_loss
                 << "\t"  << v_loss << std::endl;
        sum_loss.backward();
        optimizer.step();
        torch::save(learning_model_, MODEL_PATH);
        torch::load(nn, MODEL_PATH);
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
        std::cout << elapsedTime(start_time_) << "\t" << step_num << "\t" << sum_loss << "\t" << p_loss << "\t" << v_loss << std::endl;
        log_file << elapsedHours(start_time_) << "\t" << step_num << "\t" << sum_loss << "\t" << p_loss << "\t" << v_loss << std::endl;

        //書き出し
        learning_model_.save(MODEL_PATH);
        nn->load(MODEL_PATH);
#endif
        if (step_num % EVALUATION_INTERVAL == 0) {
            int32_t num = 0;
            float policy_loss = 0.0, value_loss = 0.0;
            for (int32_t i = 0; (i + 1) * BATCH_SIZE <= VALIDATION_SIZE; i++, num++) {
                std::tie(inputs, policy_teachers, value_teachers) = getBatch(validation_data, i * BATCH_SIZE, BATCH_SIZE);
                auto val_loss = nn->loss(inputs, policy_teachers, value_teachers);
#ifdef USE_LIBTORCH
                policy_loss += val_loss.first.item<float>();
                value_loss  += val_loss.second.item<float>();
#else
                policy_loss += val_loss.first.to_float();
                value_loss  += val_loss.second.to_float();
#endif
            }
            policy_loss /= num;
            value_loss /= num;

            validation_log << step_num << "\t" << elapsedHours(start_time_) << "\t"
                           << POLICY_LOSS_COEFF * policy_loss + VALUE_LOSS_COEFF * value_loss
                           << "\t" << policy_loss << "\t" << value_loss << std::endl;
        }

        generator.gpu_mutex.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    usi_option.stop_signal = true;
    gen_thread.detach();

    std::cout << "finish alphaZero()" << std::endl;
}