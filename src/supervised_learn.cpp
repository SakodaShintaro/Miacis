#include"learn.hpp"
#include"hyperparameter_manager.hpp"
#include<iostream>
#include<random>

void supervisedLearn() {
    HyperparameterManager settings;
    settings.add("learn_rate",          0.0f, 100.0f);
    settings.add("momentum",            0.0f, 1.0f);
    settings.add("weight_decay",        0.0f, 100.0f);
    settings.add("batch_size",          1, (int64_t)1e10);
    settings.add("data_augmentation",   0, (int64_t)1);
    settings.add("max_step",            1, (int64_t)1e10);
    settings.add("validation_interval", 1, (int64_t)1e10);
    settings.add("lr_decay_step1",      1, (int64_t)1e10);
    settings.add("lr_decay_step2",      1, (int64_t)1e10);
    settings.add("train_kifu_path");
    settings.add("valid_kifu_path");
    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
        settings.add(LOSS_TYPE_NAME[i] + "_loss_coeff", 0.0f, 1e10f);
    }

    //設定をファイルからロード
    settings.load("supervised_learn_settings.txt");

    //値の取得
    float learn_rate            = settings.get<float>("learn_rate");
    float momentum              = settings.get<float>("momentum");
    float weight_decay          = settings.get<float>("weight_decay");
    bool data_augmentation      = settings.get<int64_t>("data_augmentation");
    int64_t batch_size          = settings.get<int64_t>("batch_size");
    int64_t max_step            = settings.get<int64_t>("max_step");
    int64_t validation_interval = settings.get<int64_t>("validation_interval");
    int64_t lr_decay_step1      = settings.get<int64_t>("lr_decay_step1");
    int64_t lr_decay_step2      = settings.get<int64_t>("lr_decay_step2");
    std::string train_kifu_path = settings.get<std::string>("train_kifu_path");
    std::string valid_kifu_path = settings.get<std::string>("valid_kifu_path");

    std::array<float, LOSS_TYPE_NUM> coefficients{};
    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
        coefficients[i] = settings.get<float>(LOSS_TYPE_NAME[i] + "_loss_coeff");
    }

    //データを取得
    std::vector<LearningData> train_data = loadData(train_kifu_path);
    std::vector<LearningData> valid_data = loadData(valid_kifu_path);
    std::cout << "train_data_size = " << train_data.size() << ", valid_data_size = " << valid_data.size() << std::endl;

    //データをシャッフルするためのengine
    std::mt19937_64 engine(0);

    //学習推移のログファイル
    std::ofstream learn_log("supervised_learn_log.txt");
    dout(std::cout, learn_log) << std::fixed << "time\tepoch\tstep\t";
    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
        dout(std::cout, learn_log) << LOSS_TYPE_NAME[i] + "_loss" << "\t\n"[i == LOSS_TYPE_NUM - 1];
    }

    //validation結果のログファイル
    std::ofstream validation_log("supervised_learn_validation_log.txt");
    validation_log << std::fixed << "time\tepoch\tstep\t";
    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
        validation_log << LOSS_TYPE_NAME[i] + "_loss" << "\t\n"[i == LOSS_TYPE_NUM - 1];
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

        for (uint64_t step = 0; (step + 1) * batch_size <= train_data.size() && global_step < max_step; step++) {
            //バッチサイズ分データを確保
            std::vector<LearningData> curr_data;
            for (int64_t b = 0; b < batch_size; b++) {
                curr_data.push_back(train_data[step * batch_size + b]);
            }

            //学習
            optimizer.zero_grad();
            std::array<torch::Tensor, LOSS_TYPE_NUM> loss = neural_network->loss(curr_data, data_augmentation);
            torch::Tensor loss_sum = torch::zeros({batch_size});
            for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
                loss_sum += coefficients[i] * loss[i].cpu();
            }
            loss_sum.mean().backward();
            optimizer.step();
            global_step++;

            //表示
            dout(std::cout, learn_log) << elapsedTime(start_time) << "\t" << epoch << "\t" << global_step << "\t";
            for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
                dout(std::cout, learn_log) << loss[i].mean().item<float>() << "\t\n"[i == LOSS_TYPE_NUM - 1];
            }

            if (global_step % validation_interval == 0) {
                //validation_lossを計算
                neural_network->eval();
                std::array<float, LOSS_TYPE_NUM> valid_loss = validation(neural_network, valid_data, 4096);
                neural_network->train();
                float sum_loss = 0;
                for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
                    sum_loss += coefficients[i] * valid_loss[i];
                }

                //表示
                dout(std::cout, validation_log) << elapsedTime(start_time) << "\t" << epoch << "\t" << global_step << "\t";
                for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
                    dout(std::cout, validation_log) << valid_loss[i] << "\t\n"[i == LOSS_TYPE_NUM - 1];
                }

                //学習中のパラメータを書き出す
                torch::save(neural_network, NeuralNetworkImpl::MODEL_PREFIX + std::to_string(global_step) + ".model");
            }

            if (global_step == lr_decay_step1 || global_step == lr_decay_step2) {
                optimizer.options.learning_rate_ /= 10;
            }
        }
    }

    std::cout << "finish supervisedLearn" << std::endl;
}