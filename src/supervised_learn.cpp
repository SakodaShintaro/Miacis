﻿#include"learn.hpp"
#include"game.hpp"
#include"hyperparameter_manager.hpp"
#include<fstream>
#include<iostream>
#include<cassert>

void supervisedLearn() {
    HyperparameterManager settings;
    settings.add("learn_rate",        0.0f, 100.0f);
    settings.add("momentum",          0.0f, 1.0f);
    settings.add("weight_decay",      0.0f, 100.0f);
    settings.add("policy_loss_coeff", 0.0f, 1e10f);
    settings.add("value_loss_coeff",  0.0f, 1e10f);
    settings.add("batch_size",        1, (int64_t)1e10);
    settings.add("max_epoch",         1, (int64_t)1e10);
    settings.add("lr_decay_epoch1",   1, (int64_t)1e10);
    settings.add("lr_decay_epoch2",   1, (int64_t)1e10);
    settings.add("kifu_path");

    //設定をファイルからロード
    settings.load("supervised_learn_settings.txt");

    //値の取得
    float learn_rate        = settings.get<float>("learn_rate");
    float momentum          = settings.get<float>("momentum");
    float weight_decay      = settings.get<float>("weight_decay");
    float policy_loss_coeff = settings.get<float>("policy_loss_coeff");
    float value_loss_coeff  = settings.get<float>("value_loss_coeff");
    int64_t batch_size      = settings.get<int64_t>("batch_size");
    int64_t max_epoch       = settings.get<int64_t>("max_epoch");
    int64_t lr_decay_epoch1 = settings.get<int64_t>("lr_decay_epoch1");
    int64_t lr_decay_epoch2 = settings.get<int64_t>("lr_decay_epoch2");
    std::string kifu_path   = settings.get<std::string>("kifu_path");

    //学習データを取得
    std::vector<LearningData> data_buffer = loadData(kifu_path);

    //データをシャッフル
    std::mt19937_64 engine(0);
    std::shuffle(data_buffer.begin(), data_buffer.end(), engine);

    //validationデータを確保
    auto validation_size = (int32_t)(data_buffer.size() * 0.1) / batch_size * batch_size;
    std::vector<LearningData> validation_data(data_buffer.end() - validation_size, data_buffer.end());
    data_buffer.erase(data_buffer.end() - validation_size, data_buffer.end());
    std::cout << "learn_data_size = " << data_buffer.size() << ", validation_data_size = " << validation_size << std::endl;

    //最小値を更新した場合のみパラメータを保存する
    float min_loss = INT_MAX;

    //学習推移のログファイル
    std::ofstream learn_log("supervised_learn_log.txt");
    dout(std::cout, learn_log) << "time\tepoch\tstep\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;

    //validation結果のログファイル
    std::ofstream validation_log("supervised_learn_validation_log.txt");
    validation_log << "time\tepoch\tsum_loss\tpolicy_loss\tvalue_loss\tlearning_rate" << std::fixed << std::endl;

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

    //学習開始時間の設定
    auto start_time = std::chrono::steady_clock::now();

    //学習開始
    for (int32_t epoch = 1; epoch <= max_epoch; epoch++) {
        //データをシャッフル
        std::shuffle(data_buffer.begin(), data_buffer.end(), engine);

        for (uint64_t step = 0; (step + 1) * batch_size <= data_buffer.size(); step++) {
            //バッチサイズ分データを確保
            std::vector<LearningData> curr_data;
            for (int32_t b = 0; b < batch_size; b++) {
                curr_data.push_back(data_buffer[step * batch_size + b]);
            }

            //学習
            optimizer.zero_grad();
            auto loss = learning_model->loss(curr_data);
            auto loss_sum = policy_loss_coeff * loss[POLICY_LOSS_INDEX] + value_loss_coeff * loss[VALUE_LOSS_INDEX];
            loss_sum.mean().backward();
            optimizer.step();

            //1エポックにつき10回出力
            if ((step + 1) % (data_buffer.size() / batch_size / 10) == 0) {
                dout(std::cout, learn_log) << elapsedTime(start_time) << "\t"
                                           << epoch << "\t"
                                           << step + 1 << "\t"
                                           << loss[POLICY_LOSS_INDEX].mean().item<float>() << "\t"
                                           << loss[VALUE_LOSS_INDEX].mean().item<float>() << std::endl;
            }
        }

        //学習中のパラメータを書き出して推論用のモデルで読み込む
        torch::save(learning_model, NeuralNetworkImpl::DEFAULT_MODEL_NAME);
        torch::load(nn, NeuralNetworkImpl::DEFAULT_MODEL_NAME);

        //validation_lossを計算
        auto val_loss = validation(validation_data);
        float sum_loss = policy_loss_coeff * val_loss[POLICY_LOSS_INDEX] + value_loss_coeff * val_loss[VALUE_LOSS_INDEX];

        //最小値を更新した場合のみbest.modelとしてパラメータを保存する
        if (sum_loss < min_loss) {
            min_loss = sum_loss;
            torch::save(learning_model, NeuralNetworkImpl::MODEL_PREFIX + "_supervised_best.model");
        }

        dout(std::cout, validation_log) << elapsedTime(start_time) << "\t"
                                        << epoch << "\t"
                                        << sum_loss << "\t"
                                        << val_loss[POLICY_LOSS_INDEX] << "\t"
                                        << val_loss[VALUE_LOSS_INDEX] << "\t"
                                        << optimizer.options.learning_rate_ << std::endl;

        if (epoch == lr_decay_epoch1 || epoch == lr_decay_epoch2) {
            optimizer.options.learning_rate_ /= 10;
        }
    }

    std::cout << "finish supervisedLearn" << std::endl;
}