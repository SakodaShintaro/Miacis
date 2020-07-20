#include"learn.hpp"
#include"mcts_net.hpp"
#include"../../learn.hpp"
#include"../../hyperparameter_loader.hpp"
#include"../../common.hpp"
#include<iostream>
#include<random>

void learnMCTSNet() {
    SearchOptions options;
    HyperparameterLoader settings("learn_mcts_net_settings.txt");
    float learn_rate            = settings.get<float>("learn_rate");
    float min_learn_rate        = settings.get<float>("min_learn_rate");
    float momentum              = settings.get<float>("momentum");
    float weight_decay          = settings.get<float>("weight_decay");
    bool data_augmentation      = settings.get<bool>("data_augmentation");
    bool pretrain               = settings.get<bool>("pretrain");
    int64_t batch_size          = settings.get<int64_t>("batch_size");
    options.search_limit        = settings.get<int64_t>("search_limit");
    int64_t max_step            = settings.get<int64_t>("max_step");
    int64_t validation_interval = settings.get<int64_t>("validation_interval");
    int64_t lr_decay_mode       = settings.get<int64_t>("lr_decay_mode");
    int64_t lr_decay_step1      = settings.get<int64_t>("lr_decay_step1");
    int64_t lr_decay_step2      = settings.get<int64_t>("lr_decay_step2");
    int64_t lr_decay_period     = settings.get<int64_t>("lr_decay_period");
    std::string train_kifu_path = settings.get<std::string>("train_kifu_path");
    std::string valid_kifu_path = settings.get<std::string>("valid_kifu_path");

    //pretrain
    if (pretrain) {
        pretrainMCTSNet();
    }

    //データを取得
    std::vector<LearningData> train_data = loadData(train_kifu_path, data_augmentation);
    std::vector<LearningData> valid_data = loadData(valid_kifu_path, false);
    std::cout << "train_data_size = " << train_data.size() << ", valid_data_size = " << valid_data.size() << std::endl;

    //学習推移のログファイル
    std::ofstream train_log("mcts_net_train_log.txt");
    std::ofstream valid_log("mcts_net_valid_log.txt");
    tout(std::cout, train_log, valid_log) << std::fixed << "time\tepoch\tstep\tloss" << std::endl;

    //評価関数読み込み
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

        for (uint64_t step = 0; (step + 1) * batch_size <= train_data.size() && global_step < max_step; step++) {
            //バッチサイズ分データを確保
            std::vector<LearningData> curr_data;
            for (int64_t b = 0; b < batch_size; b++) {
                curr_data.push_back(train_data[step * batch_size + b]);
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
                torch::NoGradGuard no_grad_guard;
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

void pretrainMCTSNet() {
    HyperparameterLoader settings("learn_mcts_net_pretrain_settings.txt");
    float learn_rate            = settings.get<float>("learn_rate");
    float min_learn_rate        = settings.get<float>("min_learn_rate");
    float momentum              = settings.get<float>("momentum");
    float weight_decay          = settings.get<float>("weight_decay");
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
    std::ofstream train_log("mcts_net_pretrain_train_log.txt");
    std::ofstream valid_log("mcts_net_pretrain_valid_log.txt");
    tout(std::cout, train_log, valid_log) << std::fixed << "time\tepoch\tstep\treadout_loss\tsimulation_loss" << std::endl;

    //評価関数読み込み
    MCTSNet mcts_net;
    torch::load(mcts_net, MCTSNetImpl::DEFAULT_MODEL_NAME);
    mcts_net->setGPU(0);

    //学習前のパラメータを出力
    torch::save(mcts_net, MCTSNetImpl::MODEL_PREFIX + "_before_pretrain.model");

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

        for (uint64_t step = 0; (step + 1) * batch_size <= train_data.size() && global_step < max_step; step++) {
            //バッチサイズ分データを確保
            std::vector<LearningData> curr_data;
            for (int64_t b = 0; b < batch_size; b++) {
                curr_data.push_back(train_data[step * batch_size + b]);
            }

            //学習
            optimizer.zero_grad();
            auto[readout_loss, simulation_loss] = mcts_net->pretrainLoss(curr_data);
            readout_loss = readout_loss.mean();
            simulation_loss = simulation_loss.mean();
            torch::Tensor loss = readout_loss + simulation_loss;
            loss.backward();
            optimizer.step();
            global_step++;

            //表示
            dout(std::cout, train_log) << elapsedTime(start_time) << "\t"
                                       << epoch << "\t"
                                       << global_step << "\t"
                                       << readout_loss.item<float>() << "\t"
                                       << simulation_loss.item<float>() << "\r" << std::flush;

            if (global_step % validation_interval == 0) {
                //validation_lossを計算
                mcts_net->eval();
                torch::NoGradGuard no_grad_guard;
                std::vector<torch::Tensor> readout_losses, simulation_losses;
                for (uint64_t i = 0; i * batch_size < valid_data.size(); i++) {
                    std::vector<LearningData> curr_valid_batch;
                    for (int64_t b = 0; b < batch_size && (i * batch_size + b) < valid_data.size(); b++) {
                        curr_valid_batch.push_back(train_data[i * batch_size + b]);
                    }
                    auto[readout_loss, simulation_loss] = mcts_net->pretrainLoss(curr_valid_batch);
                    readout_losses.push_back(readout_loss.cpu());
                    simulation_losses.push_back(simulation_loss.cpu());
                }
                mcts_net->train();

                readout_loss = torch::cat(readout_losses).mean();
                simulation_loss = torch::cat(simulation_losses).mean();

                //表示
                dout(std::cout, valid_log) << elapsedTime(start_time) << "\t"
                                           << epoch << "\t"
                                           << global_step << "\t"
                                           << readout_loss.item<float>() << "\t"
                                           << simulation_loss.item<float>() << std::endl;
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

    //学習したパラメータを書き出す
    torch::save(mcts_net, MCTSNetImpl::DEFAULT_MODEL_NAME);

    std::cout << "finish pretrainMCTSNet" << std::endl;
}