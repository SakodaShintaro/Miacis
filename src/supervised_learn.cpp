#include"learn.hpp"
#include"game.hpp"
#include"operate_params.hpp"
#include"hyperparameter_manager.hpp"
#include<fstream>
#include<iostream>
#include<cassert>

void supervisedLearn() {
    HyperparameterManager settings;
    settings.add("learn_rate",             0.0f, 100.0f);
    settings.add("momentum",               0.0f, 1.0f);
    settings.add("learn_rate_decay",       0.0f, 1.0f);
    settings.add("policy_loss_coeff",      0.0f, 1e10f);
    settings.add("value_loss_coeff",       0.0f, 1e10f);
    settings.add("batch_size",             1, (int64_t)1e10);
    settings.add("patience_limit",         1, (int64_t)1e10);
    settings.add("kifu_path");

    //設定をファイルからロード
    settings.load("supervised_learn_settings.txt");
    if (!settings.check()) {
        exit(1);
    }

    //値の取得
    float learn_rate        = settings.get<float>("learn_rate");
    float momentum          = settings.get<float>("momentum");
    float learn_rate_decay  = settings.get<float>("learn_rate_decay");
    float policy_loss_coeff = settings.get<float>("policy_loss_coeff");
    float value_loss_coeff  = settings.get<float>("value_loss_coeff");
    int64_t batch_size      = settings.get<int64_t>("batch_size");
    int64_t patience_limit  = settings.get<int64_t>("patience_limit");
    std::string kifu_path   = settings.get<std::string>("kifu_path");

    //学習データを取得
    std::vector<std::pair<std::string, TeacherType>> data_buffer = loadData(kifu_path);

    //データをシャッフル
    std::mt19937_64 engine(0);
    std::shuffle(data_buffer.begin(), data_buffer.end(), engine);

    //validationデータを確保
    auto validation_size = (int32_t)(data_buffer.size() * 0.1) / batch_size * batch_size;
    std::vector<std::pair<std::string, TeacherType>> validation_data(data_buffer.end() - validation_size, data_buffer.end());
    data_buffer.erase(data_buffer.end() - validation_size, data_buffer.end());
    std::cout << "learn_data_size = " << data_buffer.size() << ", validation_data_size = " << validation_size << std::endl;

    //早期終了用の変数宣言
    float min_loss = INT_MAX;
    int32_t patience = 0;

    //学習推移のログファイル
    std::ofstream learn_log("supervised_learn_log.txt");
    learn_log << "time\tepoch\tstep\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;
    std::cout << "time\tepoch\tstep\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;

    //validation結果のログファイル
    std::ofstream validation_log("supervised_learn_validation_log.txt");
    validation_log << "time\tepoch\tsum_loss\tpolicy_loss\tvalue_loss\tpatience\tlearning_rate" << std::fixed << std::endl;

    //評価関数読み込み
    NeuralNetwork learning_model;
    torch::load(learning_model, MODEL_PATH);

    //学習前のパラメータを出力
    torch::save(learning_model, MODEL_PREFIX + "_before_learn.model");

    //optimizerの準備
    torch::optim::SGDOptions sgd_option(learn_rate);
    sgd_option.momentum(momentum);
    torch::optim::SGD optimizer(learning_model->parameters(), sgd_option);

    //学習開始時間の設定
    auto start_time = std::chrono::steady_clock::now();

    //学習開始
    for (int32_t epoch = 1; patience < patience_limit; epoch++) {
        //データをシャッフル
        std::shuffle(data_buffer.begin(), data_buffer.end(), engine);

        for (int32_t step = 0; (step + 1) * batch_size <= data_buffer.size(); step++) {
            //バッチサイズ分データを確保
            Position pos;
            std::vector<float> inputs;
            std::vector<PolicyTeacherType> policy_teachers;
            std::vector<ValueTeacherType> value_teachers;
            for (int32_t b = 0; b < batch_size; b++) {
                const auto& datum = data_buffer[step * batch_size + b];
                pos.loadSFEN(datum.first);
                const auto feature = pos.makeFeature();
                inputs.insert(inputs.end(), feature.begin(), feature.end());
                policy_teachers.push_back(datum.second.policy);
                value_teachers.push_back(datum.second.value);
            }

            //学習
            optimizer.zero_grad();
            auto loss = learning_model->loss(inputs, policy_teachers, value_teachers);
            auto loss_sum = policy_loss_coeff * loss.first + value_loss_coeff * loss.second;
            loss_sum.backward();
            optimizer.step();

            //1エポックにつき10回出力
            if ((step + 1) % (data_buffer.size() / batch_size / 10) == 0) {
                dout(std::cout, learn_log) << elapsedTime(start_time) << "\t"
                                           << epoch << "\t"
                                           << step + 1 << "\t"
                                           << loss.first.item<float>() << "\t"
                                           << loss.second.item<float>() << std::endl;
            }
        }

        //学習中のパラメータを書き出して推論用のモデルで読み込む
        torch::save(learning_model, MODEL_PATH);
        torch::load(nn, MODEL_PATH);

        //validation_lossを計算
        auto val_loss = validation(validation_data);
        float sum_loss = policy_loss_coeff * val_loss[0] + value_loss_coeff * val_loss[1];

        //validation_lossからpatience等を更新
        if (sum_loss < min_loss) {
            min_loss = sum_loss;
            patience = 0;
            torch::save(learning_model, MODEL_PREFIX + "_supervised_best.model");
        } else {
            patience++;
            optimizer.options.learning_rate_ *= learn_rate_decay;
        }

        dout(std::cout, validation_log) << elapsedTime(start_time) << "\t"
                                        << epoch << "\t"
                                        << sum_loss << "\t"
                                        << val_loss[0] << "\t"
                                        << val_loss[1] << "\t"
                                        << patience << "\t"
                                        << optimizer.options.learning_rate_ << std::endl;
    }

    std::cout << "finish supervisedLearn" << std::endl;
}