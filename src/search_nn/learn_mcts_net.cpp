#include"learn.hpp"
#include"mcts_net.hpp"
#include"../learn.hpp"
#include"../hyperparameter_loader.hpp"
#include"../common.hpp"
#include<iostream>
#include<random>

void learnMCTSNet() {
    HyperparameterLoader settings("learn_mcts_net_settings.txt");
    float learn_rate            = settings.get<float>("learn_rate");
    float min_learn_rate        = settings.get<float>("min_learn_rate");
    float momentum              = settings.get<float>("momentum");
    float weight_decay          = settings.get<float>("weight_decay");
    float mixup_alpha           = settings.get<float>("mixup_alpha");
    bool data_augmentation      = settings.get<bool>("data_augmentation");
    int64_t batch_size          = settings.get<int64_t>("batch_size");
    int64_t max_step            = settings.get<int64_t>("max_step");
    int64_t validation_interval = settings.get<int64_t>("validation_interval");
    int64_t lr_decay_mode       = settings.get<int64_t>("lr_decay_mode");
    int64_t lr_decay_step1      = settings.get<int64_t>("lr_decay_step1");
    int64_t lr_decay_step2      = settings.get<int64_t>("lr_decay_step2");
    int64_t lr_decay_period     = settings.get<int64_t>("lr_decay_period");
    std::string train_kifu_path = settings.get<std::string>("train_kifu_path");
    std::string valid_kifu_path = settings.get<std::string>("valid_kifu_path");

    //データを取得
    std::vector<LearningData> train_data = loadData(train_kifu_path, data_augmentation);
    std::vector<LearningData> valid_data = loadData(valid_kifu_path, false);
    std::cout << "train_data_size = " << train_data.size() << ", valid_data_size = " << valid_data.size() << std::endl;

    //学習推移のログファイル
    std::ofstream train_log("mcts_net_train_log.txt");
    std::ofstream valid_log("mcts_net_valid_log.txt");
    tout(std::cout, train_log, valid_log) << std::fixed << "time\tepoch\tstep\tloss" << std::endl;

    //評価関数読み込み
    SearchOptions options;
    options.search_limit = 100;
    MCTSNet mcts_net(options);
    torch::load(mcts_net, MCTSNetImpl::DEFAULT_MODEL_NAME);
    mcts_net->setGPU(0);

    //学習前のパラメータを出力
    torch::save(mcts_net, MCTSNetImpl::MODEL_PREFIX + "_before_learn.model");

    //optimizerの準備
    torch::optim::SGDOptions sgd_option(learn_rate);
    sgd_option.momentum(momentum);
    sgd_option.weight_decay(weight_decay);
    torch::optim::SGD optimizer(mcts_net->parameters(), sgd_option);

    //エポックを超えたステップ数を初期化
    int64_t global_step = 0;

    //学習開始時間の設定
    auto start_time = std::chrono::steady_clock::now();

    //学習開始
    for (int64_t epoch = 1; global_step < max_step; epoch++) {
        //データをシャッフル
        std::shuffle(train_data.begin(), train_data.end(), engine);

        for (uint64_t step = 0; (step + 1 + (mixup_alpha != 0)) * batch_size <= train_data.size() && global_step < max_step; step++) {
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
            torch::Tensor loss = mcts_net->loss(curr_data);
            loss.mean().backward();
            optimizer.step();
            global_step++;

            //表示
            dout(std::cout, train_log) << elapsedTime(start_time) << "\t"
                                       << epoch << "\t"
                                       << global_step << "\t"
                                       << loss.item<float>() << "\r" << std::flush;

            if (global_step % validation_interval == 0) {
                //validation_lossを計算
                mcts_net->eval();
                float sum_loss = 0;
                for (const LearningData& datum : valid_data) {
                    torch::Tensor valid_loss = mcts_net->loss({ datum });
                    sum_loss += valid_loss.item<float>();
                }
                sum_loss /= valid_data.size();
                mcts_net->train();

                //表示
                dout(std::cout, valid_log) << elapsedTime(start_time) << "\t"
                                           << epoch << "\t"
                                           << global_step << "\t"
                                           << sum_loss << std::endl;

                //学習中のパラメータを書き出す
                torch::save(mcts_net, MCTSNetImpl::MODEL_PREFIX + "_" + std::to_string(global_step) + ".model");
            }

            if (lr_decay_mode == 1) {
                if (global_step == lr_decay_step1 || global_step == lr_decay_step2) {
                    (dynamic_cast<torch::optim::SGDOptions&>(optimizer.param_groups().front().options())).lr() /= 10;
                }
            } else if (lr_decay_mode == 2) {
                int64_t curr_step = (step + 1) % lr_decay_period;
                (dynamic_cast<torch::optim::SGDOptions&>(optimizer.param_groups().front().options())).lr()
                        = min_learn_rate + 0.5 * (learn_rate - min_learn_rate) * (1 + cos(acos(-1) * curr_step / lr_decay_period));
            }
        }
    }

    std::cout << "finish learnMCTSNet" << std::endl;
}