﻿#include"learn.hpp"
#include"game.hpp"
#include"hyperparameter_manager.hpp"
#include<sstream>
#include<iomanip>

std::string elapsedTime(const std::chrono::steady_clock::time_point& start) {
    auto elapsed = std::chrono::steady_clock::now() - start;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();

    std::stringstream ss;

    //hh:mm:ssで文字列化
    auto minutes = seconds / 60;
    seconds %= 60;
    auto hours = minutes / 60;
    minutes %= 60;
    ss << std::setfill('0') << std::setw(3) << hours << ":"
       << std::setfill('0') << std::setw(2) << minutes << ":"
       << std::setfill('0') << std::setw(2) << seconds;
    return ss.str();
}

double elapsedHours(const std::chrono::steady_clock::time_point& start) {
    auto elapsed = std::chrono::steady_clock::now() - start;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    return seconds / 3600.0;
}

std::array<float, LOSS_TYPE_NUM> validation(const std::vector<LearningData>& data, int32_t batch_size) {
    int32_t index = 0;
    float policy_loss = 0.0, value_loss = 0.0, trans_loss = 0.0;
    torch::NoGradGuard no_grad_guard;
    while (index < data.size()) {
        std::vector<LearningData> curr_data;

        //バッチサイズ分データを確保
        while (index < data.size() && curr_data.size() < batch_size) {
            curr_data.push_back(data[index++]);
        }

        //計算
        auto loss = nn->loss(curr_data);

        policy_loss += loss[POLICY_LOSS_INDEX].sum().item<float>();
        value_loss += loss[VALUE_LOSS_INDEX].sum().item<float>();
        trans_loss += loss[TRANS_LOSS_INDEX].sum().item<float>();
    }

    //平均を求める
    policy_loss /= data.size();
    value_loss /= data.size();
    trans_loss /= data.size();

    return { policy_loss, value_loss, trans_loss };
}

std::vector<LearningData> loadData(const std::string& file_path) {
    //棋譜を読み込めるだけ読み込む
    auto games = loadGames(file_path, 100000);

    //データを局面単位にバラす
    std::vector<LearningData> data;
    for (const auto& game : games) {
        Position pos;
        for (const auto& e : game.elements) {
            LearningData datum;
            datum.SFEN = pos.toSFEN();
            datum.move = e.move;
#ifdef USE_CATEGORICAL
            datum.value = valueToIndex((pos.color() == BLACK ? game.result : MAX_SCORE + MIN_SCORE - game.result));
#else
            datum.value = (float) (pos.color() == BLACK ? game.result : MAX_SCORE + MIN_SCORE - game.result);
#endif
            data.push_back(datum);
            pos.doMove(e.move);
        }
    }

    return data;
}

void searchLearningRate() {
    HyperparameterManager settings;
    settings.add("learn_rate",             0.0f, 100.0f);
    settings.add("momentum",               0.0f, 1.0f);
    settings.add("learn_rate_decay",       0.0f, 1.0f);
    settings.add("policy_loss_coeff",      0.0f, 1e10f);
    settings.add("value_loss_coeff",       0.0f, 1e10f);
    settings.add("batch_size",             1, (int64_t)1e10);
    settings.add("patience_limit",         1, (int64_t)1e10);
    settings.add("kifu_path");

    //設定をファイルからロード.教師あり学習のものを流用する
    settings.load("supervised_learn_settings.txt");

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
    auto data_buffer = loadData(kifu_path);

    //データをシャッフル
    std::mt19937_64 engine(0);
    std::shuffle(data_buffer.begin(), data_buffer.end(), engine);

    std::cout << "learn_data_size = " << data_buffer.size() << std::endl;

    //学習率推移
    std::vector<float> lrs;

    //損失推移
    std::vector<double> losses;

    //試行回数
    constexpr int64_t times = 100;

    //学習率を上げていく倍率
    constexpr double scale = 1.2;

    for (int64_t k = 0; k < times; k++) {
        std::shuffle(data_buffer.begin(), data_buffer.end(), engine);

        //評価関数準備
        NeuralNetwork learning_model;
        torch::load(learning_model, MODEL_PATH);

        //optimizerの準備.学習率を小さい値から開始
        torch::optim::SGDOptions sgd_option(1e-5);
        sgd_option.momentum(momentum);
        torch::optim::SGD optimizer(learning_model->parameters(), sgd_option);

        for (int32_t step = 0; (step + 1) * batch_size <= data_buffer.size() && optimizer.options.learning_rate_ <= 1; step++) {
            //バッチサイズ分データを確保
            Position pos;
            std::vector<LearningData> curr_data;
            for (int32_t b = 0; b < batch_size; b++) {
                const auto& datum = data_buffer[step * batch_size + b];
                curr_data.push_back(datum);
            }

            //学習
            optimizer.zero_grad();
            auto loss = learning_model->loss(curr_data);
            auto loss_sum = policy_loss_coeff * loss[POLICY_LOSS_INDEX] + value_loss_coeff * loss[VALUE_LOSS_INDEX] + 1.0 * loss[TRANS_LOSS_INDEX];
            loss_sum.backward();
            optimizer.step();

            if (k == 0) {
                lrs.push_back(optimizer.options.learning_rate_);
                losses.push_back(loss_sum.item<float>());
            } else {
                losses[step] += loss_sum.item<float>();
            }

            optimizer.options.learning_rate_ *= scale;
        }

        std::cout << std::setw(4) << k + 1 << " / " << times << " 回終了" << std::endl;
    }

    std::cout << "学習率\t損失" << std::fixed << std::endl;
    for (int64_t i = 0; i < lrs.size(); i++) {
        std::cout << lrs[i] << "\t" << losses[i] / times << std::endl;
    }

    std::cout << "最適学習率 = " << lrs[std::min_element(losses.begin(), losses.end()) - losses.begin()] << std::endl;
}

void initParams() {
    torch::save(nn, MODEL_PATH);
    std::cout << "初期化したパラメータを" << MODEL_PATH << "に出力" << std::endl;
}