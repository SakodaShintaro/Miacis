#include"learn.hpp"
#include"hyperparameter_manager.hpp"
#include<iostream>

void supervisedLearn() {
    HyperparameterManager settings;
    settings.add("learn_rate",          0.0f, 100.0f);
    settings.add("momentum",            0.0f, 1.0f);
    settings.add("weight_decay",        0.0f, 100.0f);
    settings.add("batch_size",          1, (int64_t)1e10);
    settings.add("max_step",            1, (int64_t)1e10);
    settings.add("validation_interval", 1, (int64_t)1e10);
    settings.add("train_kifu_path");
    settings.add("valid_kifu_path");

    //設定をファイルからロード
    settings.load("supervised_learn_settings.txt");

    //値の取得
    float learn_rate            = settings.get<float>("learn_rate");
    float momentum              = settings.get<float>("momentum");
    float weight_decay          = settings.get<float>("weight_decay");
    int64_t batch_size          = settings.get<int64_t>("batch_size");
    int64_t max_step            = settings.get<int64_t>("max_step");
    int64_t validation_interval = settings.get<int64_t>("validation_interval");
    std::string train_kifu_path = settings.get<std::string>("train_kifu_path");
    std::string valid_kifu_path = settings.get<std::string>("valid_kifu_path");

    //データを取得
    std::vector<LearningData> train_data = loadData(train_kifu_path);
    std::vector<LearningData> valid_data = loadData(valid_kifu_path);
    std::cout << "train_data_size = " << train_data.size() << ", valid_data_size = " << valid_data.size() << std::endl;

    //データをシャッフルするためのengine
    std::mt19937_64 engine(0);

    //学習推移のログファイル
    std::ofstream learn_log("supervised_learn_log.txt");
    dout(std::cout, learn_log) << std::fixed << "time epoch step ";
    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
        dout(std::cout, learn_log) << std::to_string((i + 3) / STANDARD_LOSS_TYPE_NUM) + LOSS_TYPE_NAME[i % STANDARD_LOSS_TYPE_NUM] + "_loss" << " \n"[i == LOSS_TYPE_NUM - 1];
    }

    //validation結果のログファイル
    std::ofstream validation_log("supervised_learn_validation_log.txt");
    dout(std::cout, learn_log) << std::fixed << "time epoch step ";
    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
        validation_log << std::to_string((i + 3) / STANDARD_LOSS_TYPE_NUM) + LOSS_TYPE_NAME[i % STANDARD_LOSS_TYPE_NUM] + "_loss" << " \n"[i == LOSS_TYPE_NUM - 1];
    }

    //評価関数読み込み
    NeuralNetwork learning_model;
    torch::load(learning_model, NeuralNetworkImpl::DEFAULT_MODEL_NAME);

    //学習前のパラメータを出力
    torch::save(learning_model, NeuralNetworkImpl::MODEL_PREFIX + "_before_learn.model");

    //optimizerの準備
    torch::optim::SGDOptions sgd_option(learn_rate);
    sgd_option.momentum(momentum);
    sgd_option.weight_decay(weight_decay);
    torch::optim::SGD optimizer(learning_model->parameters(), sgd_option);

    //累計学習ステップ数
    int64_t sum_of_training_step = 0;

    //学習開始時間の設定
    auto start_time = std::chrono::steady_clock::now();

    //学習開始
    for (int64_t epoch = 1; epoch <= INT_MAX; epoch++) {
        //データをシャッフル
        std::shuffle(train_data.begin(), train_data.end(), engine);

        for (uint64_t step = 0; (step + 1) * batch_size <= train_data.size(); step++) {
            //バッチサイズ分データを確保
            std::vector<LearningData> curr_data;
            for (int64_t b = 0; b < batch_size; b++) {
                curr_data.push_back(train_data[step * batch_size + b]);
            }

            //学習
            optimizer.zero_grad();
            std::array<torch::Tensor, LOSS_TYPE_NUM> loss = learning_model->loss(curr_data);
            torch::Tensor loss_sum = torch::zeros({batch_size});
            for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
                loss_sum += loss[i].cpu();
            }
            loss_sum.mean().backward();
            optimizer.step();
            sum_of_training_step++;

            //表示
            dout(std::cout, learn_log) << elapsedTime(start_time) << " " << epoch << " " << sum_of_training_step << " ";
            for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
                dout(std::cout, learn_log) << loss[i].mean().item<float>() << " \n"[i == LOSS_TYPE_NUM - 1];
            }

            //validation
            if (sum_of_training_step % validation_interval == 0) {
                //学習中のパラメータを書き出して推論用のモデルで読み込む
                torch::save(learning_model, NeuralNetworkImpl::DEFAULT_MODEL_NAME);
                torch::load(nn, NeuralNetworkImpl::DEFAULT_MODEL_NAME);

                //ステップごとに別名でもパラメータを保存
                torch::save(learning_model, NeuralNetworkImpl::MODEL_PREFIX + "_" + std::to_string(sum_of_training_step) + ".model");

                //検証損失を計算
                std::array<float, LOSS_TYPE_NUM> valid_loss = validation(valid_data);

                //表示
                dout(std::cout, validation_log) << elapsedTime(start_time) << " " << epoch << " " << sum_of_training_step << " ";
                for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
                    dout(std::cout, validation_log) << valid_loss[i] << " \n"[i == LOSS_TYPE_NUM - 1];
                }
            }

            if (sum_of_training_step == max_step) {
                epoch = INT_MAX;
                break;
            }
        }
    }

    std::cout << "finish supervisedLearn" << std::endl;
}