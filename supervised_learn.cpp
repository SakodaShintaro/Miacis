#include"learn.hpp"
#include"game.hpp"
#include"operate_params.hpp"
#include<fstream>
#include<iostream>
#include<cassert>

void supervisedLearn() {
    //学習開始時間の設定
    auto start_time = std::chrono::steady_clock::now();

    //設定ファイルの読み込み
    std::ifstream ifs("supervised_learn_settings.txt");
    if (!ifs) {
        std::cerr << "fail to open supervised_learn_settings.txt" << std::endl;
        assert(false);
    }

    //後で読み込みの成功を確認するために不適当な値で初期化
    std::string kifu_path;
    int64_t batch_size = -1;
    float learn_rate = -1;
    float momentum = -1;
    float weight_decay = -1;
    int64_t patience_limit = -1;

    std::string name;
    while (ifs >> name) {
        if (name == "kifu_path") {
            ifs >> kifu_path;
        } else if (name == "batch_size") {
            ifs >> batch_size;
        } else if (name == "learn_rate") {
            ifs >> learn_rate;
        } else if (name == "momentum") {
            ifs >> momentum;
        } else if (name == "weight_decay") {
            ifs >> weight_decay;
        } else if (name == "patience") {
            ifs >> patience_limit;
        }
    }

    //読み込みの確認
    assert(batch_size > 0);
    assert(learn_rate >= 0);
    assert(momentum >= 0);
    assert(weight_decay >= 0);
    assert(patience_limit > 0);

    //学習データを取得
    std::vector<std::pair<std::string, TeacherType>> data_buffer = loadData(kifu_path);

    //データをシャッフル
    std::default_random_engine engine(0);
    std::shuffle(data_buffer.begin(), data_buffer.end(), engine);

    //validationデータを確保
    std::vector<std::pair<std::string, TeacherType>> validation_data;
    auto validation_size = (int32_t)(data_buffer.size() * 0.1) / batch_size * batch_size;
    assert(validation_size != 0);
    for (int32_t i = 0; i < validation_size; i++) {
        validation_data.push_back(data_buffer.back());
        data_buffer.pop_back();
    }
    std::cout << "learn_data_size = " << data_buffer.size() << ", validation_data_size = " << validation_size << std::endl;

    //validation用の変数宣言
    float min_loss = 1e10;
    int32_t patience = 0;

    //学習推移のログファイル
    std::ofstream learn_log("supervised_learn_log.txt", std::ios::out);
    learn_log << "time\tepoch\tstep\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;
    std::cout << "time\tepoch\tstep\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;

    //validation結果のログファイル
    std::ofstream validation_log("supervised_learn_validation_log.txt");
    validation_log << "epoch\ttime\tsum_loss\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;

    //評価関数読み込み,optimizerの準備
    NeuralNetwork learning_model;
    torch::load(learning_model, MODEL_PATH);
    torch::save(learning_model, MODEL_PREFIX + "_before_learn.model");
    torch::optim::SGDOptions sgd_option(learn_rate);
    sgd_option.momentum(momentum);
    sgd_option.weight_decay(weight_decay);
    torch::optim::SGD optimizer(learning_model->parameters(), sgd_option);

    //学習開始
    for (int32_t epoch = 1; epoch <= 1000; epoch++) {
        //データをシャッフル
        std::shuffle(data_buffer.begin(), data_buffer.end(), engine);

        for (int32_t step = 0; (step + 1) * batch_size <= data_buffer.size(); step++) {
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

            optimizer.zero_grad();
            auto loss = learning_model->loss(inputs, policy_teachers, value_teachers);
            if (step % (data_buffer.size() / batch_size / 10) == 0) {
                auto p_loss = loss.first.item<float>();
                auto v_loss = loss.second.item<float>();
                std::cout << elapsedTime(start_time)  << "\t" << epoch << "\t" << step << "\t" << p_loss << "\t" << v_loss << std::endl;
                learn_log << elapsedHours(start_time) << "\t" << epoch << "\t" << step << "\t" << p_loss << "\t" << v_loss << std::endl;
            }
            (loss.first + loss.second).backward();
            optimizer.step();
        }

        //学習中のパラメータを書き出して推論用のモデルで読み込む
        torch::save(learning_model, MODEL_PATH);
        torch::load(nn, MODEL_PATH);
        auto val_loss = validation(validation_data);
        float sum_loss = val_loss[0] + val_loss[1];

        std::cout      << epoch << "\t" << elapsedTime(start_time)  << "\t" << sum_loss << "\t" << val_loss[0] << "\t" << val_loss[1] << std::endl;
        validation_log << epoch << "\t" << elapsedHours(start_time) << "\t" << sum_loss << "\t" << val_loss[0] << "\t" << val_loss[1] << std::endl;

        if (sum_loss < min_loss) {
            min_loss = sum_loss;
            patience = 0;
            torch::save(learning_model, MODEL_PREFIX + "_supervised_best.model");
        } else if (++patience >= patience_limit) {
            break;
        } else {
            optimizer.options.learning_rate_ /= 2;
        }
    }

    std::cout << "finish SupervisedLearn" << std::endl;
}