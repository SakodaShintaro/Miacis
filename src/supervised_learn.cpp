#include "common.hpp"
#include "hyperparameter_loader.hpp"
#include "learn.hpp"
#include <iostream>
#include <random>

void supervisedLearn() {
    // clang-format off
    HyperparameterLoader settings("supervised_learn_settings.txt");
    float mixup_alpha           = settings.get<float>("mixup_alpha");
    float train_rate_threshold  = settings.get<float>("train_rate_threshold");
    bool data_augmentation      = settings.get<bool>("data_augmentation");
    bool load_multi_dir         = settings.get<bool>("load_multi_dir");
    int64_t batch_size          = settings.get<int64_t>("batch_size");
    int64_t max_step            = settings.get<int64_t>("max_step");
    std::string train_kifu_path = settings.get<std::string>("train_kifu_path");
    // clang-format on

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

    //学習クラスを生成
    LearnManager learn_manager("supervised");

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

            learn_manager.learnOneStep(curr_data, ++global_step);
        }

        if (load_multi_dir) {
            train_data = loadData(dir_paths[epoch % dir_paths.size()], data_augmentation, train_rate_threshold);
        }
    }

    std::cout << "finish supervisedLearn" << std::endl;
}