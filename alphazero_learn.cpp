#include"learn.hpp"
#include"replay_buffer.hpp"
#include"usi_options.hpp"
#include"game_generator.hpp"
#include"operate_params.hpp"
#include<thread>
#include<climits>

void alphaZero() {
    auto start_time = std::chrono::steady_clock::now();

    //オプションをファイルから読み込む
    std::ifstream ifs("alphazero_settings.txt");
    if (!ifs) {
        std::cerr << "fail to open alphazero_settings.txt" << std::endl;
        assert(false);
    }

    float learn_rate = -1;
    float learn_rate_decay = -1;
    float momentum = -1;
    float policy_loss_coeff = -1;
    float value_loss_coeff = -1;
    int64_t batch_size = -1;
    int64_t validation_interval = -1;
    int64_t max_step_num = -1;
    int64_t parallel_num = -1;
    int64_t sleep_msec = -1;
    int64_t validation_size = -1;
    std::string validation_kifu_path;

    //リプレイバッファ
    ReplayBuffer replay_buffer_;

    //ファイルを読み込んで値を設定
    std::string name;
    while (ifs >> name) {
        if (name == "batch_size") {
            ifs >> batch_size;
        } else if (name == "learn_rate") {
            ifs >> learn_rate;
        } else if (name == "learn_rate_decay") {
            ifs >> learn_rate_decay;
        } else if (name == "momentum") {
            ifs >> momentum;
        } else if (name == "random_move_num") {
            ifs >> usi_option.random_turn;
        } else if (name == "draw_turn") {
            ifs >> usi_option.draw_turn;
        } else if (name == "draw_score") {
            ifs >> usi_option.draw_score;
        } else if (name == "USI_Hash") {
            ifs >> usi_option.USI_Hash;
        } else if (name == "policy_loss_coeff") {
            ifs >> policy_loss_coeff;
        } else if (name == "value_loss_coeff") {
            ifs >> value_loss_coeff;
        } else if (name == "lambda") {
            ifs >> replay_buffer_.lambda;
        } else if (name == "max_stack_size") {
            ifs >> replay_buffer_.max_size;
        } else if (name == "first_wait") {
            ifs >> replay_buffer_.first_wait;
        } else if (name == "max_step_num") {
            ifs >> max_step_num;
        } else if (name == "playout_limit") {
            ifs >> usi_option.playout_limit;
        } else if (name == "parallel_num") {
            ifs >> parallel_num;
        } else if (name == "sleep_msec") {
            ifs >> sleep_msec;
        } else if (name == "validation_interval") {
            ifs >> validation_interval;
        } else if (name == "validation_size") {
            ifs >> validation_size;
        } else if (name == "validation_kifu_path") {
            ifs >> validation_kifu_path;
        }
    }

    //その他オプションを学習用に設定
    usi_option.limit_msec = LLONG_MAX;
    usi_option.stop_signal = false;
    usi_option.byoyomi_margin = 0LL;

    //ログファイルの設定
    std::ofstream log_file("alphazero_log.txt");
    log_file  << "time\tstep\tloss\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;
    std::cout << "time\tstep\tloss\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;
    std::ofstream validation_log("alphazero_validation_log.txt");
    validation_log << "step_num\telapsed_hours\tsum_loss\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;

    //データを取得
    std::vector<std::pair<std::string, TeacherType>> validation_data = loadData(validation_kifu_path);
    assert(validation_data.size() >= validation_size);

    //データをシャッフルして必要量以外を削除
    std::default_random_engine engine(0);
    std::shuffle(validation_data.begin(), validation_data.end(), engine);
    validation_data.erase(validation_data.begin() + validation_size);

    //モデル読み込み
#ifdef USE_LIBTORCH
    NeuralNetwork learning_model_;
    torch::load(learning_model_, MODEL_PATH);
    torch::save(learning_model_, MODEL_PREFIX + "_before_alphazero.model");
    torch::load(nn, MODEL_PATH);

    //Optimizerの準備
    torch::optim::SGDOptions sgd_option(learn_rate);
    sgd_option.momentum(momentum);
    sgd_option.weight_decay(1e-4);
    torch::optim::SGD optimizer(learning_model_->parameters(), sgd_option);

    //自己対局をしてreplay_buffer_にデータを追加するインスタンス
    GameGenerator generator(0, parallel_num, replay_buffer_, nn);
#else
    NeuralNetwork<Node> learning_model_;
    learning_model_.load(MODEL_PATH);
    learning_model_.save(MODEL_PREFIX + "_before_alphazero.model");
    nn->load(MODEL_PATH);

    O::MomentumSGD optimizer(learn_rate);
    optimizer.set_weight_decay(1e-4);
    optimizer.add(learning_model_);

    //自己対局をしてreplay_buffer_にデータを追加するインスタンス
    GameGenerator generator(0, parallel_num, replay_buffer_, *nn);

    Graph g;
    Graph::set_default(g);
#endif

    //自己対局スレッドを立てる
    std::thread gen_thread([&generator]() { generator.genGames(static_cast<int64_t>(1e10)); });

    for (int32_t step_num = 1; step_num <= max_step_num; step_num++) {
        //バッチサイズだけデータを選択
        std::vector<float> inputs;
        std::vector<PolicyTeacherType> policy_teachers;
        std::vector<ValueTeacherType> value_teachers;
        replay_buffer_.makeBatch(static_cast<int32_t>(batch_size), inputs, policy_teachers, value_teachers);

        generator.gpu_mutex.lock();
#ifdef USE_LIBTORCH
        optimizer.zero_grad();
        auto loss = learning_model_->loss(inputs, policy_teachers, value_teachers);
        auto sum_loss = policy_loss_coeff * loss.first + value_loss_coeff * loss.second;
        if (step_num % (validation_interval / 10) == 0) {
            auto p_loss = loss.first.item<float>();
            auto v_loss = loss.second.item<float>();
            std::cout << elapsedTime(start_time) << "\t" << step_num << "\t" << sum_loss.item<float>() << "\t" << p_loss
                      << "\t" << v_loss << std::endl;
            log_file << elapsedHours(start_time) << "\t" << step_num << "\t" << sum_loss.item<float>() << "\t" << p_loss
                     << "\t" << v_loss << std::endl;
        }
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
        if (step_num % (validation_interval / 10) == 0) {
            float p_loss = loss.first.to_float();
            float v_loss = loss.second.to_float();
            float sum_loss = policy_loss_coeff * p_loss + value_loss_coeff * v_loss;
            std::cout << elapsedTime(start_time) << "\t" << step_num << "\t" << sum_loss << "\t" << p_loss << "\t" << v_loss << std::endl;
            log_file << elapsedHours(start_time) << "\t" << step_num << "\t" << sum_loss << "\t" << p_loss << "\t" << v_loss << std::endl;
        }

        //書き出し
        learning_model_.save(MODEL_PATH);
        nn->load(MODEL_PATH);
#endif
        if (step_num % validation_interval == 0) {
            auto val_loss = validation(validation_data);
            std::cout      << step_num << "\t" << elapsedHours(start_time) << "\t"
                           << policy_loss_coeff * val_loss[0] + value_loss_coeff * val_loss[1] << "\t"
                           << val_loss[0] << "\t" << val_loss[1] << std::endl;
            validation_log << step_num << "\t" << elapsedHours(start_time) << "\t"
                           << policy_loss_coeff * val_loss[0] + value_loss_coeff * val_loss[1] << "\t"
                           << val_loss[0] << "\t" << val_loss[1] << std::endl;
        }

        generator.gpu_mutex.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_msec));
    }

    usi_option.stop_signal = true;
    gen_thread.detach();

    std::cout << "finish alphaZero()" << std::endl;
}