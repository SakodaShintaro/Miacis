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
    float learn_rate             = settings.get<float>("learn_rate");
    float min_learn_rate         = settings.get<float>("min_learn_rate");
    float momentum               = settings.get<float>("momentum");
    float weight_decay           = settings.get<float>("weight_decay");
    float base_policy_coeff      = settings.get<float>("base_policy_coeff");
    float base_value_coeff       = settings.get<float>("base_value_coeff");
    float entropy_coeff          = settings.get<float>("entropy_coeff");
    float train_rate_threshold   = settings.get<float>("train_rate_threshold");
    float valid_rate_threshold   = settings.get<float>("valid_rate_threshold");
    bool data_augmentation       = settings.get<bool>("data_augmentation");
    bool freeze_encoder          = settings.get<bool>("freeze_encoder");
    bool use_only_last_loss      = settings.get<bool>("use_only_last_loss");
    int64_t batch_size           = settings.get<int64_t>("batch_size");
    options.search_limit         = settings.get<int64_t>("search_limit");
    int64_t max_step             = settings.get<int64_t>("max_step");
    int64_t validation_interval  = settings.get<int64_t>("validation_interval");
    int64_t save_interval        = settings.get<int64_t>("save_interval");
    int64_t lr_decay_mode        = settings.get<int64_t>("lr_decay_mode");
    int64_t lr_decay_step1       = settings.get<int64_t>("lr_decay_step1");
    int64_t lr_decay_step2       = settings.get<int64_t>("lr_decay_step2");
    int64_t lr_decay_period      = settings.get<int64_t>("lr_decay_period");
    int64_t print_interval       = settings.get<int64_t>("print_interval");
    std::string train_kifu_path  = settings.get<std::string>("train_kifu_path");
    std::string valid_kifu_path  = settings.get<std::string>("valid_kifu_path");
    std::string encoder_path     = settings.get<std::string>("encoder_path");
    std::string policy_head_path = settings.get<std::string>("policy_head_path");
    std::string value_head_path  = settings.get<std::string>("value_head_path");
    // clang-format on

    //データを取得
    std::vector<LearningData> train_data = loadData(train_kifu_path, data_augmentation, train_rate_threshold);
    std::vector<LearningData> valid_data = loadData(valid_kifu_path, false, valid_rate_threshold);
    std::cout << "train_data_size = " << train_data.size() << ", valid_data_size = " << valid_data.size() << std::endl;

    //モデル作成
    T model(options);
    model->setGPU(0);
    model->setOption(freeze_encoder);

    //学習推移のログファイル
    std::ofstream train_log(model->modelPrefix() + "_train_log.txt");
    std::ofstream valid_log(model->modelPrefix() + "_valid_log.txt");
    tout(std::cout, train_log, valid_log) << std::fixed << "time\tepoch\tstep";
    for (int64_t i = 0; i <= options.search_limit; i++) {
        if (i % print_interval == 0) {
            tout(std::cout, train_log, valid_log) << "\tpolicy_loss_" << i << "\tvalue_loss_" << i;
        } else {
            dout(train_log, valid_log) << "\tpolicy_loss_" << i << "\tvalue_loss_" << i;
        }
    }
    tout(std::cout, train_log, valid_log) << "\tbase_policy\tbase_value\tentropy" << std::endl;
    std::cout << std::setprecision(4);

    //encoderを既存のパラメータから読み込み
    model->loadPretrain(encoder_path, policy_head_path, value_head_path);

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

            //損失計算
            optimizer.zero_grad();
            std::vector<torch::Tensor> loss = model->loss(curr_data);
            global_step++;

            //表示
            if (global_step % std::max(validation_interval / 100, (int64_t)1) == 0) {
                dout(std::cout, train_log) << elapsedTime(start_time) << "\t" << epoch << "\t" << global_step;
                for (uint64_t m = 0; m < loss.size(); m++) {
                    if (m / 2 % print_interval == 0 || m > (uint64_t)options.search_limit * 2) {
                        //標準出力にも表示
                        dout(std::cout, train_log) << "\t" << loss[m].item<float>();
                    } else {
                        //ファイルにだけ表示
                        train_log << "\t" << loss[m].item<float>();
                    }
                }

                dout(std::cout, train_log) << "\r" << std::flush;
            }

            //勾配計算,パラメータ更新
            if (use_only_last_loss) {
                //最後以外0にする
                for (int64_t i = 0; i < options.search_limit; i++) {
                    loss[i] *= 0;
                }
            } else {
                //平均化する
                for (int64_t i = 0; i <= options.search_limit; i++) {
                    loss[i] /= (options.search_limit + 1);
                }
            }
            loss[loss.size() - 3] *= base_policy_coeff;
            loss[loss.size() - 2] *= base_value_coeff;
            loss[loss.size() - 1] *= entropy_coeff;
            torch::Tensor loss_sum = torch::stack(loss).sum();
            loss_sum.mean().backward();
            optimizer.step();

            if (global_step % validation_interval == 0) {
                //validation_lossを計算
                model->eval();
                torch::NoGradGuard no_grad_guard;
                std::vector<float> valid_loss_sum(loss.size(), 0);
                for (uint64_t j = 0; j < valid_data.size(); j += batch_size) {
                    std::vector<LearningData> curr_valid_data;
                    for (int64_t b = 0; b < batch_size && j + b < valid_data.size(); b++) {
                        curr_valid_data.push_back(valid_data[j + b]);
                    }

                    std::vector<torch::Tensor> valid_loss = model->loss(curr_valid_data);
                    for (uint64_t i = 0; i < valid_loss_sum.size(); i++) {
                        valid_loss_sum[i] += valid_loss[i].item<float>() * curr_valid_data.size();
                    }
                }
                for (float& v : valid_loss_sum) {
                    v /= valid_data.size();
                }
                model->train();

                //表示
                dout(std::cout, valid_log) << elapsedTime(start_time) << "\t" << epoch << "\t" << global_step;
                for (uint64_t m = 0; m < loss.size(); m++) {
                    if (m / 2 % print_interval == 0 || m > (uint64_t)options.search_limit * 2) {
                        //標準出力にも表示
                        dout(std::cout, valid_log) << "\t" << valid_loss_sum[m];
                    } else {
                        //ファイルにだけ表示
                        valid_log << "\t" << valid_loss_sum[m];
                    }
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

    //SimpleMLPのときはencoder部分などを保存したいのでここで保存
    model->saveParts();

    std::cout << "finish learnSearchNN" << std::endl;
}

template<class T> void validSearchNN(const std::string& model_name) {
    SearchOptions options;

    //データを取得
    std::string valid_kifu_path;
    std::cout << "validation_kifu_path: ";
    std::cin >> valid_kifu_path;
    float valid_rate_threshold{};
    std::cout << "valid_rate_threshold: ";
    std::cin >> valid_rate_threshold;
    uint64_t batch_size{};
    std::cout << "batch_size: ";
    std::cin >> batch_size;
    std::cout << "search_limit: ";
    std::cin >> options.search_limit;

    std::vector<LearningData> valid_data = loadData(valid_kifu_path, false, valid_rate_threshold);
    std::cout << "valid_data_size = " << valid_data.size() << std::endl;

    //モデル作成
    T model(options);
    model->setGPU(0);
    model->setOption(true);
    torch::load(model, model->defaultModelName());
    model->eval();
    torch::NoGradGuard no_grad_guard;

    //validation_lossを計算
    std::vector<float> valid_loss_sum(options.search_limit + 1, 0);
    for (uint64_t i = 0; i < valid_data.size();) {
        std::vector<LearningData> curr_data;
        while (curr_data.size() < batch_size && i < valid_data.size()) {
            curr_data.push_back(valid_data[i++]);
        }
        std::vector<torch::Tensor> valid_loss = model->loss(curr_data);
        for (uint64_t j = 0; j < valid_loss_sum.size(); j++) {
            valid_loss_sum[j] += valid_loss[j].item<float>() * curr_data.size();
        }
        std::cout << "finish " << std::setw(4) << i << " / " << valid_data.size() << "\r";
    }
    for (float& v : valid_loss_sum) {
        v /= valid_data.size();
    }

    //表示
    std::cout << std::endl;
    for (float v : valid_loss_sum) {
        std::cout << v << "\t";
    }
    std::cout << std::endl;

    std::cout << "finish validSearchNN" << std::endl;
}

#endif //MIACIS_LEARN_SEARCH_NN_HPP