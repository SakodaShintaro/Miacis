#include"learn.hpp"
#include"replay_buffer.hpp"
#include"usi_options.hpp"
#include"game_generator.hpp"
#include"operate_params.hpp"
#include"hyperparameter_manager.hpp"
#include<thread>
#include<climits>

void alphaZero() {
    HyperparameterManager settings;
    settings.add("learn_rate",          0.0f, 100.0f);
    settings.add("learn_rate_decay",    0.0f, 1.0f);
    settings.add("momentum",            0.0f, 1.0f);
    settings.add("policy_loss_coeff",   0.0f, 1e10f);
    settings.add("value_loss_coeff",    0.0f, 1e10f);
    settings.add("lambda",              0.0f, 1.0f);
    settings.add("draw_turn",           0, (int64_t)1024);
    settings.add("random_turn",         0, (int64_t)1024);
    settings.add("batch_size",          1, (int64_t)1e10);
    settings.add("thread_num",          1, (int64_t)std::thread::hardware_concurrency());
    settings.add("max_step_num",        1, (int64_t)1e10);
    settings.add("sleep_msec",          0, (int64_t)1e10);
    settings.add("max_stack_size",      1, (int64_t)1e10);
    settings.add("first_wait",          0, (int64_t)1e10);
    settings.add("search_limit",        1, (int64_t)1e10);
    settings.add("search_batch_size",   1, (int64_t)1e10);
    settings.add("validation_interval", 1, (int64_t)1e10);
    settings.add("validation_size",     0, (int64_t)1e10);
    settings.add("validation_kifu_path");

    //設定をファイルからロード
    settings.load("alphazero_settings.txt");
    if (!settings.check()) {
        exit(1);
    }

    //リプレイバッファ
    ReplayBuffer replay_buffer(settings.get<int64_t>("first_wait"), settings.get<int64_t>("max_stack_size"), settings.get<float>("lambda"));

    //usi_optionの設定
    usi_option.random_turn       = settings.get<int64_t>("random_turn");
    usi_option.draw_turn         = settings.get<int64_t>("draw_turn");
    usi_option.thread_num        = settings.get<int64_t>("thread_num");
    usi_option.search_batch_size = settings.get<int64_t>("search_batch_size");
    usi_option.search_limit      = settings.get<int64_t>("search_limit");

    //その他オプションを学習用に設定
    usi_option.limit_msec = LLONG_MAX;
    usi_option.stop_signal = false;
    usi_option.byoyomi_margin = 0LL;

    //学習ループ中で複数回参照するオプションは変数として確保する
    //速度的な問題はほぼないと思うが,学習始まってからtypoで中断が入るのも嫌なので
    uint64_t batch_size         = static_cast<uint64_t>(settings.get<int64_t>("batch_size"));
    int64_t max_step_num        = settings.get<int64_t>("max_step_num");
    int64_t validation_interval = settings.get<int64_t>("validation_interval");
    int64_t sleep_msec          = settings.get<int64_t>("sleep_msec");
    float policy_loss_coeff     = settings.get<float>("policy_loss_coeff");
    float value_loss_coeff      = settings.get<float>("value_loss_coeff");

    //ログファイルの設定
    std::ofstream learn_log("alphazero_log.txt");
    learn_log << "time\tstep\tloss\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;
    std::cout << "time\tstep\tloss\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;
    std::ofstream validation_log("alphazero_validation_log.txt");
    validation_log << "step_num\telapsed_hours\tsum_loss\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;

    //データを取得
    std::vector<std::pair<std::string, TeacherType>> validation_data = loadData(settings.get<std::string>("validation_kifu_path"));
    assert(validation_data.size() >= settings.get<int64_t>("validation_size"));

    //データをシャッフルして必要量以外を削除
    std::default_random_engine engine(0);
    std::shuffle(validation_data.begin(), validation_data.end(), engine);
    validation_data.erase(validation_data.begin() + settings.get<int64_t>("validation_size"), validation_data.end());
    validation_data.shrink_to_fit();

    //モデル読み込み
    NeuralNetwork learning_model;
    learning_model->setGPU(0);
    torch::load(learning_model, MODEL_PATH);
    torch::save(learning_model, MODEL_PREFIX + "_before_alphazero.model");
    torch::load(nn, MODEL_PATH);

    //Optimizerの準備
    torch::optim::SGDOptions sgd_option(settings.get<float>("learn_rate"));
    sgd_option.momentum(settings.get<float>("momentum"));
    torch::optim::SGD optimizer(learning_model->parameters(), sgd_option);

    //時間計測開始
    auto start_time = std::chrono::steady_clock::now();

    //自己対局スレッドを立てる
    auto gpu_num = torch::getNumGPUs();
    std::vector<NeuralNetwork> additional_nn(gpu_num - 1);
    for (uint64_t i = 0; i < gpu_num - 1; i++) {
        additional_nn[i]->setGPU(static_cast<int16_t>(i + 1));
    }

    std::vector<std::unique_ptr<GameGenerator>> generators(gpu_num);
    for (uint64_t i = 0; i < gpu_num; i++) {
        generators[i] = std::make_unique<GameGenerator>(replay_buffer, i == 0 ? nn : additional_nn[i - 1]);
    }

    std::vector<std::thread> gen_threads;
    for (uint64_t i = 0; i < gpu_num; i++) {
        gen_threads.emplace_back([&generators, i]() { generators[i]->genGames((int64_t)(1e15)); });
    }

    //入力,教師データ
    std::vector<float> inputs(batch_size * SQUARE_NUM * INPUT_CHANNEL_NUM);
    std::vector<PolicyTeacherType> policy_teachers(batch_size);
    std::vector<ValueTeacherType> value_teachers(batch_size);

    for (int32_t step_num = 1; step_num <= max_step_num; step_num++) {
        //バッチサイズだけデータを選択
        replay_buffer.makeBatch(batch_size, inputs, policy_teachers, value_teachers);

        generators.front()->gpu_mutex.lock();
        optimizer.zero_grad();
        auto loss = learning_model->loss(inputs, policy_teachers, value_teachers);
        auto sum_loss = policy_loss_coeff * loss.first + value_loss_coeff * loss.second;
        if (step_num == 1) {
            double h = elapsedHours(start_time);
            std::cout << settings.get<int64_t>("first_wait") / (h * 3600) << " pos / sec" << std::endl;
        }
        if (step_num % (validation_interval / 10) == 0) {
            dout(std::cout, learn_log) << elapsedTime(start_time) << "\t"
                                       << step_num << "\t"
                                       << sum_loss.item<float>() << "\t"
                                       << loss.first.item<float>() << "\t"
                                       << loss.second.item<float>() << std::endl;
        }
        sum_loss.backward();
        optimizer.step();
        torch::save(learning_model, MODEL_PATH);
        if (step_num % validation_interval == 0) {
            //モデルの保存,読み込みなど
            torch::load(nn, MODEL_PATH);
            for (uint64_t i = 0; i < gpu_num - 1; i++) {
                generators[i + 1]->gpu_mutex.lock();
                torch::load(additional_nn[i], MODEL_PATH);
                additional_nn[i]->setGPU(static_cast<int16_t>(i + 1));
                generators[i + 1]->gpu_mutex.unlock();
            }
            torch::save(learning_model, MODEL_PREFIX + "_" + std::to_string(step_num) + ".model");

            //validation
            auto val_loss = validation(validation_data);
            dout(std::cout, validation_log) << elapsedTime(start_time) << "\t"
                                            << step_num << "\t"
                                            << policy_loss_coeff * val_loss[0] + value_loss_coeff * val_loss[1] << "\t"
                                            << val_loss[0] << "\t"
#ifdef USE_CATEGORICAL
                                            //Categoricalのときは3つ目の損失(期待値を取って二乗誤差)がある
                                            << val_loss[1] << "\t"
                                            << val_loss[2] << std::endl;
#else
                                            << val_loss[1] << std::endl;
#endif
        }

        generators.front()->gpu_mutex.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_msec));
    }

    usi_option.stop_signal = true;
    for (auto& th : gen_threads) {
        th.join();
    }

    std::cout << "finish alphaZero()" << std::endl;
}