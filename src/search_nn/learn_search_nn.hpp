#ifndef MIACIS_LEARN_SEARCH_NN_HPP
#define MIACIS_LEARN_SEARCH_NN_HPP

#include "../common.hpp"
#include "../hyperparameter_loader.hpp"
#include "../learn.hpp"
#include <iostream>
#include <random>

template<class T> void learnSearchNN(const std::string& model_name) {
    // clang-format off
    SearchOptions options;
    HyperparameterLoader settings("learn_search_nn_settings.txt");
    float learn_rate            = settings.get<float>("learn_rate");
    float min_learn_rate        = settings.get<float>("min_learn_rate");
    float momentum              = settings.get<float>("momentum");
    float weight_decay          = settings.get<float>("weight_decay");
    bool data_augmentation      = settings.get<bool>("data_augmentation");
    int64_t batch_size          = settings.get<int64_t>("batch_size");
    options.search_limit        = settings.get<int64_t>("search_limit");
    int64_t max_step            = settings.get<int64_t>("max_step");
    int64_t validation_interval = settings.get<int64_t>("validation_interval");
    int64_t save_interval       = settings.get<int64_t>("save_interval");
    int64_t lr_decay_mode       = settings.get<int64_t>("lr_decay_mode");
    int64_t lr_decay_step1      = settings.get<int64_t>("lr_decay_step1");
    int64_t lr_decay_step2      = settings.get<int64_t>("lr_decay_step2");
    int64_t lr_decay_period     = settings.get<int64_t>("lr_decay_period");
    std::string train_kifu_path = settings.get<std::string>("train_kifu_path");
    std::string valid_kifu_path = settings.get<std::string>("valid_kifu_path");
    std::string encoder_path    = settings.get<std::string>("encoder_path");
    // clang-format on

    //データを取得
    std::vector<LearningData> train_data = loadData(train_kifu_path, data_augmentation);
    std::vector<LearningData> valid_data = loadData(valid_kifu_path, false);
    std::cout << "train_data_size = " << train_data.size() << ", valid_data_size = " << valid_data.size() << std::endl;

    //学習推移のログファイル
    std::ofstream train_log(model_name + "_train_log.txt");
    std::ofstream valid_log(model_name + "_valid_log.txt");
    tout(std::cout, train_log, valid_log) << std::fixed << "time\tepoch\tstep";
    for (int64_t i = 0; i < options.search_limit; i++) {
        tout(std::cout, train_log, valid_log) << "\tloss_" << i + 1;
    }
    tout(std::cout, train_log, valid_log) << std::endl;

    //モデル作成
    T model(options);
    model->setGPU(0);

    //encoderを既存のパラメータから読み込み
    torch::load(model->encoder, encoder_path);

    //学習前のパラメータを出力
    torch::save(model, model->modelPrefix() + "_before_learn.model");

    //optimizerの準備
    torch::optim::SGDOptions sgd_option(learn_rate);
    sgd_option.momentum(momentum);
    sgd_option.weight_decay(weight_decay);
    torch::optim::SGD optimizer(model->parameters(), sgd_option);

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
            std::vector<torch::Tensor> loss = model->loss(curr_data);
            torch::Tensor loss_sum = torch::cat(loss).sum();
            loss_sum.mean().backward();
            optimizer.step();
            global_step++;

            //表示
            dout(std::cout, train_log) << elapsedTime(start_time) << "\t" << epoch << "\t" << global_step;
            for (const torch::Tensor& t : loss) {
                dout(std::cout, train_log) << "\t" << t.item<float>();
            }
            dout(std::cout, train_log) << "\r" << std::flush;

            if (global_step % validation_interval == 0) {
                //validation_lossを計算
                model->eval();
                torch::NoGradGuard no_grad_guard;
                std::vector<float> valid_loss_sum(loss.size(), 0);
                for (const LearningData& datum : valid_data) {
                    std::vector<torch::Tensor> valid_loss = model->loss({ datum });
                    for (uint64_t i = 0; i < valid_loss_sum.size(); i++) {
                        valid_loss_sum[i] += valid_loss[i].item<float>();
                    }
                }
                for (float& v : valid_loss_sum) {
                    v /= valid_data.size();
                }
                model->train();

                //表示
                dout(std::cout, valid_log) << elapsedTime(start_time) << "\t" << epoch << "\t" << global_step;
                for (float v : valid_loss_sum) {
                    dout(std::cout, valid_log) << "\t" << v;
                }
                dout(std::cout, valid_log) << std::endl;
            }
            if (global_step % save_interval == 0) {
                //学習中のパラメータを書き出す
                torch::save(model, model->modelPrefix() + "_" + std::to_string(global_step) + ".model");
            }

            if (lr_decay_mode == 1) {
                if (global_step == lr_decay_step1 || global_step == lr_decay_step2) {
                    (dynamic_cast<torch::optim::SGDOptions&>(optimizer.param_groups().front().options())).lr() /= 10;
                }
            } else if (lr_decay_mode == 2) {
                int64_t curr_step = (step + 1) % lr_decay_period;
                (dynamic_cast<torch::optim::SGDOptions&>(optimizer.param_groups().front().options())).lr() =
                    min_learn_rate + 0.5 * (learn_rate - min_learn_rate) * (1 + cos(acos(-1) * curr_step / lr_decay_period));
            }
        }
    }

    std::cout << "finish learnSearchNN" << std::endl;
}

#endif //MIACIS_LEARN_SEARCH_NN_HPP