#include "common.hpp"
#include "hyperparameter_loader.hpp"
#include "learn.hpp"
#include <iostream>
#include <random>

void supervisedLearn() {
    // clang-format off
    HyperparameterLoader settings("supervised_learn_settings.txt");
    float learn_rate            = settings.get<float>("learn_rate");
    float min_learn_rate        = settings.get<float>("min_learn_rate");
    float momentum              = settings.get<float>("momentum");
    float weight_decay          = settings.get<float>("weight_decay");
    float mixup_alpha           = settings.get<float>("mixup_alpha");
    float train_rate_threshold  = settings.get<float>("train_rate_threshold");
    float valid_rate_threshold  = settings.get<float>("valid_rate_threshold");
    bool data_augmentation      = settings.get<bool>("data_augmentation");
    bool load_multi_dir         = settings.get<bool>("load_multi_dir");
    int64_t batch_size          = settings.get<int64_t>("batch_size");
    int64_t max_step            = settings.get<int64_t>("max_step");
    int64_t validation_interval = settings.get<int64_t>("validation_interval");
    int64_t lr_decay_mode       = settings.get<int64_t>("lr_decay_mode");
    int64_t lr_decay_step1      = settings.get<int64_t>("lr_decay_step1");
    int64_t lr_decay_step2      = settings.get<int64_t>("lr_decay_step2");
    int64_t lr_decay_step3      = settings.get<int64_t>("lr_decay_step3");
    int64_t lr_decay_period     = settings.get<int64_t>("lr_decay_period");
    std::string train_kifu_path = settings.get<std::string>("train_kifu_path");
    std::string valid_kifu_path = settings.get<std::string>("valid_kifu_path");
    // clang-format on

    std::array<float, LOSS_TYPE_NUM> coefficients{};
    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
        coefficients[i] = settings.get<float>(LOSS_TYPE_NAME[i] + "_loss_coeff");
    }

    //ディレクトリを逐次的に展開していく場合、まず展開するパス名を取得する
    std::vector<std::string> dir_paths;
    if (load_multi_dir) {
        dir_paths = childFiles(train_kifu_path);
        train_kifu_path = dir_paths[0];
    }

    //学習設定などに関するログ。現状はすぐ下のところで使っているだけ
    std::ofstream other_log("other_log.txt");

    //データを取得
    std::vector<LearningData> train_data = loadData(train_kifu_path, data_augmentation, train_rate_threshold);
    std::vector<LearningData> valid_data = loadData(valid_kifu_path, false, valid_rate_threshold);
    dout(std::cout, other_log) << "train_data_size = " << train_data.size() << ", valid_data_size = " << valid_data.size()
                               << ", dir_paths.size() = " << dir_paths.size() << std::endl;

    //学習推移のログファイル
    std::ofstream train_log("supervised_train_log.txt");
    std::ofstream valid_log("supervised_valid_log.txt");
    tout(std::cout, train_log, valid_log) << std::fixed << "time\tepoch\tstep\t";
    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
        tout(std::cout, train_log, valid_log) << LOSS_TYPE_NAME[i] + "_loss"
                                              << "\t\n"[i == LOSS_TYPE_NUM - 1];
    }

    //評価関数読み込み
    NeuralNetwork neural_network;
    torch::load(neural_network, NeuralNetworkImpl::DEFAULT_MODEL_NAME);
    neural_network->setGPU(0);

    //学習前のパラメータを出力
    torch::save(neural_network, NeuralNetworkImpl::MODEL_PREFIX + "_before_learn.model");

    //optimizerの準備
    torch::optim::SGDOptions sgd_option(learn_rate);
    sgd_option.momentum(momentum);
    sgd_option.weight_decay(weight_decay);
    torch::optim::SGD optimizer(neural_network->parameters(), sgd_option);

    //エポックを超えたステップ数を初期化
    int64_t global_step = 0;

    //学習開始時間の設定
    auto start_time = std::chrono::steady_clock::now();

    //学習開始
    for (int64_t epoch = 1; global_step < max_step; epoch++) {
        //データをシャッフル
        std::shuffle(train_data.begin(), train_data.end(), engine);

        for (uint64_t step = 0; (step + 1 + (mixup_alpha != 0)) * batch_size <= train_data.size() && global_step < max_step;
             step++) {
            //バッチサイズ分データを確保
            std::vector<LearningData> curr_data;
            for (int64_t b = 0; b < batch_size; b++) {
                curr_data.push_back(train_data[step * batch_size + b]);
            }

            if (mixup_alpha != 0) {
                //データ2つを混ぜて使うので倍の量取る
                step++;
                for (int64_t b = 0; b < batch_size; b++) {
                    curr_data.push_back(train_data[step * batch_size + b]);
                }
            }

            //学習
            optimizer.zero_grad();
            std::array<torch::Tensor, LOSS_TYPE_NUM> loss =
                (mixup_alpha == 0 ? neural_network->loss(curr_data)
                                  : neural_network->mixUpLossFinalLayer(curr_data, mixup_alpha));
            torch::Tensor loss_sum = torch::zeros({ batch_size });
            for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
                loss_sum += coefficients[i] * loss[i].cpu();
            }
            loss_sum.mean().backward();
            optimizer.step();
            global_step++;

            //表示
            dout(std::cout, train_log) << elapsedTime(start_time) << "\t" << epoch << "\t" << global_step << "\t";
            for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
                dout(std::cout, train_log) << loss[i].mean().item<float>() << "\t\r"[i == LOSS_TYPE_NUM - 1];
            }
            dout(std::cout, train_log) << std::flush;

            if (global_step % validation_interval == 0) {
                //validation_lossを計算
                neural_network->eval();
                std::array<float, LOSS_TYPE_NUM> valid_loss = validation(neural_network, valid_data, batch_size);
                neural_network->train();
                float sum_loss = 0;
                for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
                    sum_loss += coefficients[i] * valid_loss[i];
                }

                //表示
                dout(std::cout, valid_log) << elapsedTime(start_time) << "\t" << epoch << "\t" << global_step << "\t";
                for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
                    dout(std::cout, valid_log) << valid_loss[i] << "\t\n"[i == LOSS_TYPE_NUM - 1];
                }
                dout(std::cout, valid_log) << std::flush;

                //学習中のパラメータを書き出す
                torch::save(neural_network, NeuralNetworkImpl::MODEL_PREFIX + "_" + std::to_string(global_step) + ".model");
            }

            if (lr_decay_mode == 1) {
                if (global_step == lr_decay_step1 || global_step == lr_decay_step2 || global_step == lr_decay_step3) {
                    (dynamic_cast<torch::optim::SGDOptions&>(optimizer.param_groups().front().options())).lr() /= 10;
                }
            } else if (lr_decay_mode == 2) {
                int64_t curr_step = global_step % lr_decay_period;
                (dynamic_cast<torch::optim::SGDOptions&>(optimizer.param_groups().front().options())).lr() =
                    min_learn_rate + 0.5 * (learn_rate - min_learn_rate) * (1 + cos(acos(-1) * curr_step / lr_decay_period));
            }
        }

        if (load_multi_dir) {
            train_data = loadData(dir_paths[epoch % dir_paths.size()], data_augmentation, train_rate_threshold);
        }
    }

    std::cout << "finish supervisedLearn" << std::endl;
}