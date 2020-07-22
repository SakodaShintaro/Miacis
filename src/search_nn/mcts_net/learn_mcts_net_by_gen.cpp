//#include"mcts_net.hpp"
//#include"../../learn.hpp"
//#include"../../game_generator.hpp"
//#include"../../hyperparameter_loader.hpp"
//
//void learnMCTSNetByGen() {
//    SearchOptions search_options;
//    HyperparameterLoader settings("learn_mcts_net_by_gen_settings.txt");
//    float learn_rate                  = settings.get<float>("learn_rate");
//    float min_learn_rate              = settings.get<float>("min_learn_rate");
//    float momentum                    = settings.get<float>("momentum");
//    float weight_decay                = settings.get<float>("weight_decay");
//    float lambda                      = settings.get<float>("lambda");
//    float alpha                       = settings.get<float>("alpha");
//    float mixup_alpha                 = settings.get<float>("mixup_alpha");
//    float Q_dist_lambda               = settings.get<float>("Q_dist_lambda");
//    float noise_epsilon               = settings.get<float>("noise_epsilon");
//    float noise_alpha                 = settings.get<float>("noise_alpha");
//    search_options.temperature_x1000  = settings.get<float>("Q_dist_temperature") * 1000;
//    search_options.C_PUCT_x1000       = settings.get<float>("C_PUCT") * 1000;
//    search_options.use_fp16           = settings.get<bool>("use_fp16");
//    search_options.draw_turn          = settings.get<int64_t>("draw_turn");
//    search_options.random_turn        = settings.get<int64_t>("random_turn");
//    search_options.thread_num_per_gpu = settings.get<int64_t>("thread_num_per_gpu");
//    search_options.search_limit       = settings.get<int64_t>("search_limit");
//    search_options.search_batch_size  = settings.get<int64_t>("search_batch_size");
//    int64_t batch_size                = settings.get<int64_t>("batch_size");
//    int64_t max_step_num              = settings.get<int64_t>("max_step_num");
//    int64_t learn_rate_decay_mode     = settings.get<int64_t>("learn_rate_decay_mode");
//    int64_t learn_rate_decay_step1    = settings.get<int64_t>("learn_rate_decay_step1");
//    int64_t learn_rate_decay_step2    = settings.get<int64_t>("learn_rate_decay_step2");
//    int64_t learn_rate_decay_step3    = settings.get<int64_t>("learn_rate_decay_step3");
//    int64_t learn_rate_decay_period   = settings.get<int64_t>("learn_rate_decay_period");
//    int64_t update_interval           = settings.get<int64_t>("update_interval");
//    int64_t batch_size_per_gen        = settings.get<int64_t>("batch_size_per_gen");
//    int64_t worker_num_per_thread     = settings.get<int64_t>("worker_num_per_thread");
//    int64_t max_stack_size            = settings.get<int64_t>("max_stack_size");
//    int64_t first_wait                = settings.get<int64_t>("first_wait");
//    int64_t output_interval           = settings.get<int64_t>("output_interval");
//    int64_t save_interval             = settings.get<int64_t>("save_interval");
//    int64_t validation_interval       = settings.get<int64_t>("validation_interval");
//    int64_t sleep_msec                = settings.get<int64_t>("sleep_msec");
//    int64_t init_buffer_by_kifu       = settings.get<int64_t>("init_buffer_by_kifu");
//    int64_t noise_mode                = settings.get<int64_t>("noise_mode");
//    bool data_augmentation            = settings.get<bool>("data_augmentation");
//    bool Q_search                     = settings.get<bool>("Q_search");
//    std::string training_kifu_path    = settings.get<std::string>("training_kifu_path");
//    std::string validation_kifu_path  = settings.get<std::string>("validation_kifu_path");
//
//    std::array<float, LOSS_TYPE_NUM> coefficients{};
//    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
//        coefficients[i] = settings.get<float>(LOSS_TYPE_NAME[i] + "_loss_coeff");
//    }
//
//    //カテゴリカルモデルでもQをもとに探索したい場合
//    if (Q_search) {
//        search_options.P_coeff_x1000 = 0;
//        search_options.Q_coeff_x1000 = 1000;
//    }
//
//    //値同士の制約関係を確認
//    assert(save_interval % update_interval == 0);
//
//    //リプレイバッファの生成
//    ReplayBuffer replay_buffer(first_wait, max_stack_size, output_interval, lambda, alpha, data_augmentation);
//
//    if (init_buffer_by_kifu) {
//        replay_buffer.fillByKifu(training_kifu_path);
//    }
//
//    //ログファイルの設定
//    std::ofstream train_log("mcts_net_train_log.txt");
//    std::ofstream valid_log("mcts_net_valid_log.txt");
//    dout(std::cout, train_log) << "time\tstep\tsum_loss";
//    valid_log << "time\tstep\tsum_loss";
//    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
//        dout(std::cout, train_log) << "\t" + LOSS_TYPE_NAME[i] + "_loss";
//        valid_log << "\t" + LOSS_TYPE_NAME[i] + "_loss";
//    }
//    dout(std::cout, train_log) << std::fixed << std::endl;
//    valid_log << std::fixed << std::endl;
//
//    //validation用のデータを取得
//    std::vector<LearningData> validation_data = loadData(validation_kifu_path, false);
//    std::cout << "validation_data.size() = " << validation_data.size() << std::endl;
//
//    //学習に使うネットワークの生成
//    MCTSNet mcts_net(search_options);
//    torch::load(mcts_net, MCTSNetImpl::DEFAULT_MODEL_NAME);
//    mcts_net->setGPU(0);
//
//    //学習前のパラメータを保存
//    torch::save(mcts_net, MCTSNetImpl::MODEL_PREFIX + "_before_learn.model");
//
//    //Optimizerの準備
//    torch::optim::SGDOptions sgd_option(learn_rate);
//    sgd_option.momentum(momentum);
//    sgd_option.weight_decay(weight_decay);
//    torch::optim::SGD optimizer(mcts_net->parameters(), sgd_option);
//
//    //時間計測開始
//    auto start_time = std::chrono::steady_clock::now();
//
//    //GPUの数だけネットワーク,自己対局生成器を生成
//    size_t gpu_num = torch::getNumGPUs();
//    std::vector<NeuralNetwork> neural_networks(gpu_num);
//    std::vector<std::unique_ptr<GameGenerator>> generators(gpu_num);
//    std::vector<std::thread> gen_threads;
//    for (uint64_t i = 0; i < gpu_num; i++) {
//        torch::load(neural_networks[i], MCTSNetImpl::DEFAULT_MODEL_NAME);
//        neural_networks[i]->setGPU(static_cast<int16_t>(i), search_options.use_fp16);
//        generators[i] = std::make_unique<GameGenerator>(search_options, worker_num_per_thread, Q_dist_lambda,
//                                                        noise_mode, noise_epsilon, noise_alpha, replay_buffer, neural_networks[i]);
//        gen_threads.emplace_back([&generators, i]() { generators[i]->genGames(); });
//    }
//
//    //学習ループ
//    for (int64_t step_num = 1; step_num <= max_step_num; step_num++) {
//        //バッチサイズ分データを選択
//        //mixupをオンにしたとき(mixup_alpha != 0のとき)はバッチサイズの2倍取る
//        std::vector<LearningData> curr_data = replay_buffer.makeBatch(batch_size * (1 + (mixup_alpha != 0)));
//
//        //1回目はmakeBatch内で十分棋譜が貯まるまで待ち時間が発生する.その生成速度を計算
//        if (step_num == 1) {
//            double gen_speed = first_wait / (elapsedHours(start_time) * 3600);
//            if (sleep_msec == -1) {
//                sleep_msec = (int64_t) (batch_size * 1000 / (batch_size_per_gen * gen_speed));
//            }
//            std::ofstream ofs("gen_speed.txt");
//            dout(std::cout, ofs) << "gen_speed = " << gen_speed << " pos / sec, sleep_msec = " << sleep_msec << std::endl;
//        }
//
//        //学習用ネットワークは生成用ネットワークの先頭とGPUを共有しているのでロックをかける
//        generators.front()->gpu_mutex.lock();
//
//        //損失計算
//        optimizer.zero_grad();
//        torch::Tensor loss = mcts_net->loss(curr_data);
//
//        //逆伝播してパラメータ更新
//        loss.backward();
//        optimizer.step();
//
//        //表示
//        dout(std::cout, train_log) << elapsedTime(start_time) << "\t" << step_num << "\t" << loss.item<float>() << "\t\r" << std::flush;
//
//        //一定間隔でActorのパラメータをLearnerと同期
//        if (step_num % update_interval == 0) {
//            //学習パラメータを保存
//            torch::save(mcts_net, MCTSNetImpl::DEFAULT_MODEL_NAME);
//
//            //各ネットワークで保存されたパラメータを読み込み
//            for (uint64_t i = 0; i < gpu_num; i++) {
//                if (i > 0) {
//                    generators[i]->gpu_mutex.lock();
//                }
//
//                //ロードするときは一度fp32に直さないとエラーになる
//                //もっと良いやり方はありそうだがなぁ
//                neural_networks[i]->setGPU(i, false);
//                torch::load(neural_networks[i], MCTSNetImpl::DEFAULT_MODEL_NAME);
//                neural_networks[i]->setGPU(static_cast<int16_t>(i), search_options.use_fp16);
//                if (i > 0) {
//                    generators[i]->gpu_mutex.unlock();
//                }
//            }
//        }
//
//        //パラメータをステップ付きで保存
//        if (step_num % save_interval == 0) {
//            torch::save(mcts_net, MCTSNetImpl::MODEL_PREFIX + "_" + std::to_string(step_num) + ".model");
//        }
//
//        if (step_num % validation_interval == 0) {
//            mcts_net->eval();
//            float sum_loss = 0;
//            for (const LearningData& datum : validation_data) {
//                torch::Tensor valid_loss = mcts_net->loss({ datum });
//                sum_loss += valid_loss.item<float>();
//            }
//            sum_loss /= validation_data.size();
//            mcts_net->train();
//            dout(std::cout, valid_log) << elapsedTime(start_time) << "\t" << step_num << "\t" << sum_loss << "\t" << std::endl;
//        }
//
//        //学習率の減衰
//        if (learn_rate_decay_mode == 1) {
//            if (step_num == learn_rate_decay_step1
//                || step_num == learn_rate_decay_step2
//                || step_num == learn_rate_decay_step3) {
//                (dynamic_cast<torch::optim::SGDOptions&>(optimizer.param_groups().front().options())).lr() /= 10;
//            }
//        } else if (learn_rate_decay_mode == 2) {
//            int64_t curr_step = step_num % learn_rate_decay_period;
//            (dynamic_cast<torch::optim::SGDOptions&>(optimizer.param_groups().front().options())).lr()
//                    = min_learn_rate + 0.5 * (learn_rate - min_learn_rate) * (1 + cos(acos(-1) * curr_step / learn_rate_decay_period));
//        } else if (learn_rate_decay_mode == 3) {
//            if (step_num % learn_rate_decay_period == 0) {
//                (dynamic_cast<torch::optim::SGDOptions&>(optimizer.param_groups().front().options())).lr() *= 0.9;
//            }
//        }
//
//        //GPUを解放
//        generators.front()->gpu_mutex.unlock();
//
//        //学習スレッドを眠らせることで擬似的にActorの数を増やす
//        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_msec));
//    }
//
//    //生成スレッドを止める
//    for (uint64_t i = 0; i < gpu_num; i++) {
//        generators[i]->stop_signal = true;
//    }
//    for (std::thread& th : gen_threads) {
//        th.join();
//    }
//
//    std::cout << "finish learnMCTSNetByGen" << std::endl;
//}