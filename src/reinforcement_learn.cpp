﻿#include "game_generator.hpp"
#include "hyperparameter_loader.hpp"
#include "learn.hpp"

void reinforcementLearn() {
    // clang-format off
    SearchOptions search_options;
    HyperparameterLoader settings("reinforcement_learn_settings.txt");
    float lambda                      = settings.get<float>("lambda");
    float per_alpha                   = settings.get<float>("per_alpha");
    float mixup_alpha                 = settings.get<float>("mixup_alpha");
    float Q_dist_lambda               = settings.get<float>("Q_dist_lambda");
    float noise_epsilon               = settings.get<float>("noise_epsilon");
    float noise_alpha                 = settings.get<float>("noise_alpha");
    float train_rate_threshold        = settings.get<float>("train_rate_threshold");
    search_options.temperature_x1000  = settings.get<float>("Q_dist_temperature") * 1000;
    search_options.C_PUCT_x1000       = settings.get<float>("C_PUCT") * 1000;
    search_options.use_fp16           = settings.get<bool>("use_fp16");
    search_options.draw_turn          = settings.get<int64_t>("draw_turn");
    search_options.random_turn        = settings.get<int64_t>("random_turn");
    search_options.thread_num_per_gpu = settings.get<int64_t>("thread_num_per_gpu");
    search_options.search_limit       = settings.get<int64_t>("search_limit");
    search_options.search_batch_size  = settings.get<int64_t>("search_batch_size");
    int64_t batch_size                = settings.get<int64_t>("batch_size");
    int64_t max_step_num              = settings.get<int64_t>("max_step_num");
    int64_t update_interval           = settings.get<int64_t>("update_interval");
    int64_t batch_size_per_gen        = settings.get<int64_t>("batch_size_per_gen");
    int64_t worker_num_per_thread     = settings.get<int64_t>("worker_num_per_thread");
    int64_t max_stack_size            = settings.get<int64_t>("max_stack_size");
    int64_t first_wait                = settings.get<int64_t>("first_wait");
    int64_t output_interval           = settings.get<int64_t>("output_interval");
    int64_t sleep_msec                = settings.get<int64_t>("sleep_msec");
    int64_t init_buffer_by_kifu       = settings.get<int64_t>("init_buffer_by_kifu");
    int64_t noise_mode                = settings.get<int64_t>("noise_mode");
    bool data_augmentation            = settings.get<bool>("data_augmentation");
    bool Q_search                     = settings.get<bool>("Q_search");
    std::string train_kifu_path       = settings.get<std::string>("train_kifu_path");
    std::string valid_kifu_path       = settings.get<std::string>("valid_kifu_path");
    // clang-format on

    //学習クラスを生成
    LearnManager learn_manager("reinforcement");

    //カテゴリカルモデルでもQをもとに探索したい場合
    if (Q_search) {
        search_options.P_coeff_x1000 = 0;
        search_options.Q_coeff_x1000 = 1000;
    }

    //リプレイバッファの生成
    ReplayBuffer replay_buffer(first_wait, max_stack_size, output_interval, lambda, per_alpha, data_augmentation);

    if (init_buffer_by_kifu) {
        replay_buffer.fillByKifu(train_kifu_path, train_rate_threshold);
    }

    //時間計測開始
    auto start_time = std::chrono::steady_clock::now();

    //GPUの数だけネットワーク,自己対局生成器を生成
    size_t gpu_num = torch::getNumGPUs();
    std::vector<NeuralNetwork> neural_networks(gpu_num);
    std::vector<std::unique_ptr<GameGenerator>> generators(gpu_num);
    std::vector<std::thread> gen_threads;
    for (uint64_t i = 0; i < gpu_num; i++) {
        torch::load(neural_networks[i], NeuralNetworkImpl::DEFAULT_MODEL_NAME);
        neural_networks[i]->setGPU(static_cast<int16_t>(i), search_options.use_fp16);
        generators[i] = std::make_unique<GameGenerator>(search_options, worker_num_per_thread, Q_dist_lambda, noise_mode,
                                                        noise_epsilon, noise_alpha, replay_buffer, neural_networks[i]);
        gen_threads.emplace_back([&generators, i]() { generators[i]->genGames(); });
    }

    //学習ループ
    for (int64_t step_num = 1; step_num <= max_step_num; step_num++) {
        //バッチサイズ分データを選択
        //mixupをオンにしたとき(mixup_alpha_ != 0のとき)はバッチサイズの2倍取る
        std::vector<LearningData> curr_data = replay_buffer.makeBatch(batch_size * (1 + (mixup_alpha != 0)));

        //1回目はmakeBatch内で十分棋譜が貯まるまで待ち時間が発生する.その生成速度を計算
        if (step_num == 1) {
            float gen_speed = first_wait / (elapsedHours(start_time) * 3600);
            if (sleep_msec == -1) {
                sleep_msec = (int64_t)(batch_size * 1000 / (batch_size_per_gen * gen_speed));
            }
            std::ofstream ofs("gen_speed.txt");
            dout(std::cout, ofs) << "gen_speed = " << gen_speed << " pos / sec, sleep_msec = " << sleep_msec << std::endl;
        }

        //学習用ネットワークは生成用ネットワークの先頭とGPUを共有しているのでロックをかける
        generators.front()->gpu_mutex.lock();

        //1ステップ学習し、損失を取得
        torch::Tensor loss_sum = learn_manager.learnOneStep(curr_data, step_num);

        //replay_bufferのpriorityを更新
        std::vector<float> loss_vec(loss_sum.data_ptr<float>(), loss_sum.data_ptr<float>() + batch_size);

        //mixupモードのときは損失を複製して2倍に拡張。これもうちょっと上手く書けないものか……
        if (mixup_alpha != 0) {
            std::vector<float> copy = loss_vec;
            loss_vec.clear();
            for (float v : copy) {
                loss_vec.push_back(v);
                loss_vec.push_back(v);
            }
        }
        replay_buffer.update(loss_vec);

        //一定間隔でActorのパラメータをLearnerと同期
        if (step_num % update_interval == 0) {
            //学習パラメータを保存
            torch::save(learn_manager.neural_network, NeuralNetworkImpl::DEFAULT_MODEL_NAME);

            //各ネットワークで保存されたパラメータを読み込み
            for (uint64_t i = 0; i < gpu_num; i++) {
                if (i > 0) {
                    generators[i]->gpu_mutex.lock();
                }

                //ロードするときは一度fp32に直さないとエラーになる
                //もっと良いやり方はありそうだがなぁ
                neural_networks[i]->setGPU(i, false);
                torch::load(neural_networks[i], NeuralNetworkImpl::DEFAULT_MODEL_NAME);
                neural_networks[i]->setGPU(static_cast<int16_t>(i), search_options.use_fp16);
                if (i > 0) {
                    generators[i]->gpu_mutex.unlock();
                }
            }
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
    for (std::thread& th : gen_threads) {
        th.join();
    }

    std::cout << "finish reinforcementLearn" << std::endl;
}