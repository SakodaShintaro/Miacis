#include"learn.hpp"
#include"game_generator.hpp"
#include"hyperparameter_manager.hpp"

void alphaZero() {
    HyperparameterManager settings;
    settings.add("learn_rate",             0.0f, 100.0f);
    settings.add("momentum",               0.0f, 1.0f);
    settings.add("lambda",                 0.0f, 1.0f);
    settings.add("alpha",                  0.0f, 1e10f);
    settings.add("Q_dist_temperature",     0.0f, 1e10f);
    settings.add("Q_dist_lambda",          0.0f, 1.0f);
    settings.add("C_PUCT",                 0.0f, 1e10f);
    settings.add("draw_turn",              0, (int64_t)1024);
    settings.add("random_turn",            0, (int64_t)1024);
    settings.add("batch_size",             1, (int64_t)1e10);
    settings.add("thread_num",             1, (int64_t)std::thread::hardware_concurrency());
    settings.add("max_step_num",           1, (int64_t)1e10);
    settings.add("learn_rate_decay_step1", 0, (int64_t)1e10);
    settings.add("learn_rate_decay_step2", 0, (int64_t)1e10);
    settings.add("learn_rate_decay_step3", 0, (int64_t)1e10);
    settings.add("update_interval",        1, (int64_t)1e10);
    settings.add("batch_size_per_gen",     1, (int64_t)1e10);
    settings.add("worker_num_per_gpu",     1, (int64_t)1e10);
    settings.add("max_stack_size",         1, (int64_t)1e10);
    settings.add("first_wait",             0, (int64_t)1e10);
    settings.add("search_limit",           1, (int64_t)1e10);
    settings.add("search_batch_size",      1, (int64_t)1e10);
    settings.add("output_interval",        1, (int64_t)1e10);
    settings.add("save_interval",          1, (int64_t)1e10);
    settings.add("validation_interval",    1, (int64_t)1e10);
    settings.add("validation_kifu_path");
    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
        settings.add(LOSS_TYPE_NAME[i] + "_loss_coeff", 0.0f, 1e10f);
    }

    //設定をファイルからロード
    settings.load("alphazero_settings.txt");

    //値の取得
    SearchOptions search_options;
    float learn_rate                 = settings.get<float>("learn_rate");
    float momentum                   = settings.get<float>("momentum");
    float lambda                     = settings.get<float>("lambda");
    float alpha                      = settings.get<float>("alpha");
    float Q_dist_lambda              = settings.get<float>("Q_dist_lambda");
    search_options.temperature_x1000 = settings.get<float>("Q_dist_temperature") * 1000;
    search_options.C_PUCT_x1000      = settings.get<float>("C_PUCT") * 1000;
    search_options.draw_turn         = settings.get<int64_t>("draw_turn");
    search_options.random_turn       = settings.get<int64_t>("random_turn");
    search_options.thread_num        = settings.get<int64_t>("thread_num");
    search_options.search_limit      = settings.get<int64_t>("search_limit");
    search_options.search_batch_size = settings.get<int64_t>("search_batch_size");
    int64_t batch_size               = settings.get<int64_t>("batch_size");
    int64_t max_step_num             = settings.get<int64_t>("max_step_num");
    int64_t learn_rate_decay_step1   = settings.get<int64_t>("learn_rate_decay_step1");
    int64_t learn_rate_decay_step2   = settings.get<int64_t>("learn_rate_decay_step2");
    int64_t learn_rate_decay_step3   = settings.get<int64_t>("learn_rate_decay_step3");
    int64_t update_interval          = settings.get<int64_t>("update_interval");
    int64_t batch_size_per_gen       = settings.get<int64_t>("batch_size_per_gen");
    int64_t worker_num_per_gpu       = settings.get<int64_t>("worker_num_per_gpu");
    int64_t max_stack_size           = settings.get<int64_t>("max_stack_size");
    int64_t first_wait               = settings.get<int64_t>("first_wait");
    int64_t output_interval          = settings.get<int64_t>("output_interval");
    int64_t save_interval            = settings.get<int64_t>("save_interval");
    int64_t validation_interval      = settings.get<int64_t>("validation_interval");
    std::string validation_kifu_path = settings.get<std::string>("validation_kifu_path");

    std::array<float, LOSS_TYPE_NUM> coefficients{};
    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
        coefficients[i] = settings.get<float>(LOSS_TYPE_NAME[i] + "_loss_coeff");
    }

    //値同士の制約関係を確認
    assert(save_interval % update_interval == 0);

    //学習スレッドをスリープさせる時間は生成速度から自動計算するので不適な値で初期化
    int64_t sleep_msec = -1;

    //リプレイバッファの生成
    ReplayBuffer replay_buffer(first_wait, max_stack_size, output_interval, lambda, alpha);

    //ログファイルの設定
    std::ofstream learn_log("alphazero_log.txt");
    std::ofstream validation_log("alphazero_validation_log.txt");
    dout(std::cout, learn_log) << "time\tstep\tsum_loss";
    validation_log             << "time\tstep\tsum_loss";
    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
        dout(std::cout, learn_log) << "\t" + LOSS_TYPE_NAME[i] + "_loss";
        validation_log             << "\t" + LOSS_TYPE_NAME[i] + "_loss";
    }
    dout(std::cout, learn_log) << std::fixed << std::endl;
    validation_log             << std::fixed << std::endl;

    //データを取得
    std::vector<LearningData> validation_data = loadData(validation_kifu_path);
    std::cout << "validation_data.size() = " << validation_data.size() << std::endl;

    //モデル読み込み
    NeuralNetwork learning_model;
    torch::load(learning_model, NeuralNetworkImpl::DEFAULT_MODEL_NAME);
    learning_model->setGPU(0);

    //学習前のパラメータを保存
    torch::save(learning_model, NeuralNetworkImpl::MODEL_PREFIX + "_before_alphazero.model");

    //Optimizerの準備
    torch::optim::SGDOptions sgd_option(learn_rate);
    sgd_option.momentum(momentum);
    torch::optim::SGD optimizer(learning_model->parameters(), sgd_option);

    //時間計測開始
    auto start_time = std::chrono::steady_clock::now();

    //GPUの数だけネットワーク,自己対局生成器を生成
    size_t gpu_num = torch::getNumGPUs();
    std::vector<NeuralNetwork> neural_networks(gpu_num);
    std::vector<std::unique_ptr<GameGenerator>> generators(gpu_num);
    std::vector<std::thread> gen_threads;
    for (uint64_t i = 0; i < gpu_num; i++) {
        torch::load(neural_networks[i], NeuralNetworkImpl::DEFAULT_MODEL_NAME);
        neural_networks[i]->setGPU(static_cast<int16_t>(i));
        generators[i] = std::make_unique<GameGenerator>(search_options, worker_num_per_gpu, Q_dist_lambda, replay_buffer, neural_networks[i]);
        gen_threads.emplace_back([&generators, i]() { generators[i]->genGames(); });
    }

    for (int64_t step_num = 1; step_num <= max_step_num; step_num++) {
        //バッチサイズ分データを選択
        std::vector<LearningData> curr_data = replay_buffer.makeBatch(batch_size);

        //1回目はmakeBatch内で十分棋譜が貯まるまで待ち時間が発生する.その生成速度を計算
        if (step_num == 1) {
            std::ofstream ofs("gen_speed.txt");
            double gen_speed = first_wait / (elapsedHours(start_time) * 3600);
            sleep_msec = (int64_t)(batch_size * 1000 / (batch_size_per_gen * gen_speed));
            dout(std::cout, ofs) << "gen_speed = " << gen_speed << " pos / sec, sleep_msec = " << sleep_msec << std::endl;
        }

        //先頭のネットワークとはGPUを共有しているのでロックをかける
        generators.front()->gpu_mutex.lock();

        //損失計算
        optimizer.zero_grad();
        std::array<torch::Tensor, LOSS_TYPE_NUM> loss = learning_model->loss(curr_data);
        torch::Tensor loss_sum = torch::zeros({batch_size});
        for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
            loss_sum += coefficients[i] * loss[i].cpu();
        }

        //replay_bufferのpriorityを更新
        std::vector<float> loss_vec(loss_sum.data<float>(), loss_sum.data<float>() + batch_size);
        replay_buffer.update(loss_vec);

        //損失をバッチについて平均を取ったものに修正
        loss_sum = loss_sum.mean();

        //学習
        loss_sum.backward();
        optimizer.step();
        torch::save(learning_model, NeuralNetworkImpl::DEFAULT_MODEL_NAME);

        //1回パラメータ保存する間隔につき10回表示
        if (step_num % (save_interval / 10) == 0) {
            dout(std::cout, learn_log) << elapsedTime(start_time) << "\t" << step_num << "\t" << loss_sum.item<float>() << "\t";
            for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
                dout(std::cout, learn_log) << loss[i].mean().item<float>() << "\t\n"[i == LOSS_TYPE_NUM - 1];
            }
        }

        //一定間隔でモデルを読み込んでActorのパラメータをLearnerと同期
        if (step_num % update_interval == 0) {
            for (uint64_t i = 0; i < gpu_num; i++) {
                if (i > 0) {
                    generators[i]->gpu_mutex.lock();
                }
                torch::load(neural_networks[i], NeuralNetworkImpl::DEFAULT_MODEL_NAME);
                neural_networks[i]->setGPU(static_cast<int16_t>(i));
                if (i > 0) {
                    generators[i]->gpu_mutex.unlock();
                }
            }
        }

        //パラメータをステップ付きで保存
        if (step_num % save_interval == 0) {
            torch::save(learning_model, NeuralNetworkImpl::MODEL_PREFIX + "_" + std::to_string(step_num) + ".model");
        }

        if (step_num % validation_interval == 0) {
            learning_model->eval();
            std::array<float, LOSS_TYPE_NUM> valid_loss = validation(learning_model, validation_data, 4096);
            learning_model->train();
            float valid_loss_sum = 0.0;
            for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
                valid_loss_sum += coefficients[i] * valid_loss[i];
            }
            dout(std::cout, validation_log) << elapsedTime(start_time) << "\t" << step_num << "\t" << valid_loss_sum << "\t";
            for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
                dout(std::cout, validation_log) << valid_loss[i] << "\t\n"[i == LOSS_TYPE_NUM - 1];
            }
        }

        //学習率の減衰.AlphaZeroを意識して3回まで設定可能
        if (step_num == learn_rate_decay_step1
         || step_num == learn_rate_decay_step2
         || step_num == learn_rate_decay_step3) {
            optimizer.options.learning_rate_ /= 10;
        }

        //GPUを解放
        generators.front()->gpu_mutex.unlock();

        //学習スレッドを眠らせることで擬似的にActorの数を増やす
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_msec));
    }

    //生成スレッドを止める
    for (uint64_t i = 0; i < gpu_num; i++) {
        generators[i]->stop_signal = true;
    }
    for (auto& th : gen_threads) {
        th.join();
    }

    std::cout << "finish alphaZero" << std::endl;
}