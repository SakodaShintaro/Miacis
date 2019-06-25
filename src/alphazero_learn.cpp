#include"learn.hpp"
#include"replay_buffer.hpp"
#include"game_generator.hpp"
#include"hyperparameter_manager.hpp"
#include<thread>
#include<climits>

void alphaZero() {
    HyperparameterManager settings;
    settings.add("learn_rate",             0.0f, 100.0f);
    settings.add("momentum",               0.0f, 1.0f);
    settings.add("policy_loss_coeff",      0.0f, 1e10f);
    settings.add("value_loss_coeff",       0.0f, 1e10f);
    settings.add("lambda",                 0.0f, 1.0f);
    settings.add("alpha",                  0.0f, 1e10f);
    settings.add("learn_rate_decay_step1", 0, (int64_t)1e10);
    settings.add("learn_rate_decay_step2", 0, (int64_t)1e10);
    settings.add("learn_rate_decay_step3", 0, (int64_t)1e10);
    settings.add("draw_turn",              0, (int64_t)1024);
    settings.add("batch_size",             1, (int64_t)1e10);
    settings.add("thread_num",             1, (int64_t)std::thread::hardware_concurrency());
    settings.add("max_step_num",           1, (int64_t)1e10);
    settings.add("update_interval",        1, (int64_t)1e10);
    settings.add("batch_size_per_gen",     1, (int64_t)1e10);
    settings.add("max_stack_size",         1, (int64_t)1e10);
    settings.add("first_wait",             0, (int64_t)1e10);
    settings.add("search_limit",           1, (int64_t)1e10);
    settings.add("search_batch_size",      1, (int64_t)1e10);
    settings.add("validation_interval",    1, (int64_t)1e10);
    settings.add("validation_size",        0, (int64_t)1e10);
    settings.add("validation_kifu_path");

    //設定をファイルからロード
    settings.load("alphazero_settings.txt");

    //値の取得
    float learn_rate                 = settings.get<float>("learn_rate");
    float momentum                   = settings.get<float>("momentum");
    float policy_loss_coeff          = settings.get<float>("policy_loss_coeff");
    float value_loss_coeff           = settings.get<float>("value_loss_coeff");
    float lambda                     = settings.get<float>("lambda");
    float alpha                      = settings.get<float>("alpha");
    int64_t learn_rate_decay_step1   = settings.get<int64_t>("learn_rate_decay_step1");
    int64_t learn_rate_decay_step2   = settings.get<int64_t>("learn_rate_decay_step2");
    int64_t learn_rate_decay_step3   = settings.get<int64_t>("learn_rate_decay_step3");
    int64_t draw_turn                = settings.get<int64_t>("draw_turn");
    int64_t batch_size               = settings.get<int64_t>("batch_size");
    int64_t thread_num               = settings.get<int64_t>("thread_num");
    int64_t max_step_num             = settings.get<int64_t>("max_step_num");
    int64_t update_interval          = settings.get<int64_t>("update_interval");
    int64_t batch_size_per_gen       = settings.get<int64_t>("batch_size_per_gen");
    int64_t max_stack_size           = settings.get<int64_t>("max_stack_size");
    int64_t first_wait               = settings.get<int64_t>("first_wait");
    int64_t search_limit             = settings.get<int64_t>("search_limit");
    int64_t search_batch_size        = settings.get<int64_t>("search_batch_size");
    int64_t validation_interval      = settings.get<int64_t>("validation_interval");
    int64_t validation_size          = settings.get<int64_t>("validation_size");
    std::string validation_kifu_path = settings.get<std::string>("validation_kifu_path");

    //値同士の制約関係を確認
    assert(validation_interval % update_interval == 0);

    //学習スレッドをスリープさせる時間は生成速度から自動計算するので初期化は不要だが一応
    int64_t sleep_msec = 1000;

    //探索のシグナルをオフにしておく
    Searcher::stop_signal = false;

    //リプレイバッファの生成
    ReplayBuffer replay_buffer(first_wait, max_stack_size, lambda, alpha);

    //ログファイルの設定
    std::ofstream learn_log("alphazero_log.txt");
    std::ofstream validation_log("alphazero_validation_log.txt");
    dout(std::cout, learn_log) << "time\tstep\tsum_loss\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;
    validation_log             << "time\tstep\tsum_loss\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;

    //データを取得
    std::vector<std::pair<std::string, TeacherType>> validation_data = loadData(validation_kifu_path);
    assert(validation_data.size() >= validation_size);

    //データをシャッフルして必要量以外を削除
    std::mt19937_64 engine(0);
    std::shuffle(validation_data.begin(), validation_data.end(), engine);
    validation_data.erase(validation_data.begin() + validation_size, validation_data.end());
    validation_data.shrink_to_fit();

    //モデル読み込み
    NeuralNetwork learning_model;
    learning_model->setGPU(0);
    torch::load(learning_model, MODEL_PATH);
    torch::load(nn, MODEL_PATH);

    //学習前のパラメータを保存
    torch::save(learning_model, MODEL_PREFIX + "_before_alphazero.model");

    //Optimizerの準備
    torch::optim::SGDOptions sgd_option(learn_rate);
    sgd_option.momentum(momentum);
    torch::optim::SGD optimizer(learning_model->parameters(), sgd_option);

    //時間計測開始
    auto start_time = std::chrono::steady_clock::now();

    //GPUの数だけネットワークを生成
    auto gpu_num = torch::getNumGPUs();
    std::vector<NeuralNetwork> additional_nn(gpu_num - 1);
    for (uint64_t i = 0; i < gpu_num - 1; i++) {
        additional_nn[i]->setGPU(static_cast<int16_t>(i + 1));
    }

    //自己対局スレッドを生成.0番目のものはnnを使い、それ以外は上で生成したネットワークを使う
    std::vector<std::unique_ptr<GameGenerator>> generators(gpu_num);
    for (uint64_t i = 0; i < gpu_num; i++) {
        generators[i] = std::make_unique<GameGenerator>(search_limit, draw_turn, thread_num, search_batch_size,
                                                        replay_buffer, i == 0 ? nn : additional_nn[i - 1]);
    }

    //生成開始.10^15個の(つまり無限に)棋譜を生成させる
    std::vector<std::thread> gen_threads;
    for (uint64_t i = 0; i < gpu_num; i++) {
        gen_threads.emplace_back([&generators, i]() { generators[i]->genGames((int64_t)(1e15)); });
    }

    //入力,教師データ
    std::vector<float> inputs(batch_size * SQUARE_NUM * INPUT_CHANNEL_NUM);
    std::vector<PolicyTeacherType> policy_teachers(batch_size);
    std::vector<ValueTeacherType> value_teachers(batch_size);

    for (int32_t step_num = 1; step_num <= max_step_num; step_num++) {
        //バッチサイズ分データを選択
        replay_buffer.makeBatch(batch_size, inputs, policy_teachers, value_teachers);

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
        auto loss = learning_model->loss(inputs, policy_teachers, value_teachers);
        auto loss_sum = (policy_loss_coeff * loss.first + value_loss_coeff * loss.second).cpu();

        //replay_bufferのpriorityを更新
        std::vector<float> loss_vec(loss_sum.data<float>(), loss_sum.data<float>() + batch_size);
        replay_buffer.update(loss_vec);

        //損失をバッチについて平均を取ったものに修正
        loss_sum = loss_sum.mean();

        //学習
        loss_sum.backward();
        optimizer.step();
        torch::save(learning_model, MODEL_PATH);

        //1回のvalidationまで10回表示
        if (step_num % (validation_interval / 10) == 0) {
            dout(std::cout, learn_log) << elapsedTime(start_time) << "\t"
                                       << step_num << "\t"
                                       << loss_sum.item<float>() << "\t"
                                       << loss.first.mean().item<float>() << "\t"
                                       << loss.second.mean().item<float>() << std::endl;
        }

        //一定間隔でモデルの読み込んでActorのパラメータをLearnerと同期
        if (step_num % update_interval == 0) {
            torch::load(nn, MODEL_PATH);
            for (uint64_t i = 0; i < gpu_num - 1; i++) {
                generators[i + 1]->gpu_mutex.lock();
                torch::load(additional_nn[i], MODEL_PATH);
                additional_nn[i]->setGPU(static_cast<int16_t>(i + 1));
                generators[i + 1]->gpu_mutex.unlock();
            }
        }

        //validation
        if (step_num % validation_interval == 0) {
            //パラメータをステップ付きで保存
            torch::save(learning_model, MODEL_PREFIX + "_" + std::to_string(step_num) + ".model");

            auto val_loss = validation(validation_data);
            dout(std::cout, validation_log) << elapsedTime(start_time) << "\t"
                                            << step_num << "\t"
                                            << policy_loss_coeff * val_loss[0] + value_loss_coeff * val_loss[1] << "\t"
                                            << val_loss[0] << "\t"
                                            << val_loss[1] << std::endl;
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
    Searcher::stop_signal = true;
    for (auto& th : gen_threads) {
        th.join();
    }

    std::cout << "finish alphaZero" << std::endl;
}