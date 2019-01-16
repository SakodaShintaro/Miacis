﻿#include"bonanza_method_trainer.hpp"
#include"position.hpp"
#include"MCTSearcher.hpp"
#include"usi_options.hpp"
#include<chrono>
#include<vector>
#include<algorithm>
#include<utility>
#include<functional>
#include<iostream>
#include<iomanip>
#include<thread>

BonanzaMethodTrainer::BonanzaMethodTrainer(std::string settings_file_path) {
    std::ifstream ifs(settings_file_path);
    if (!ifs) {
        std::cerr << "fail to open setting_file(" << settings_file_path << ")" << std::endl;
        assert(false);
    }

    std::string name;
    KIFU_PATH = "/home/sakoda/Downloads/2016";
    while (ifs >> name) {
        if (name == "kifu_path") {
            ifs >> KIFU_PATH;
        } else if (name == "game_num") {
            ifs >> game_num_;
        } else if (name == "batch_size") {
            ifs >> BATCH_SIZE;
        } else if (name == "L1_grad_coefficient") {
            ifs >> L1_GRAD_COEFFICIENT;
        } else if (name == "optimizer") {
            ifs >> OPTIMIZER_NAME;
            if (!isLegalOptimizer()) {
                std::cerr << "Optimizerは[SGD, AdaGrad, RMSProp, AdaDelta]から選択" << std::endl;
                assert(false);
            }
        } else if (name == "learn_rate") {
            ifs >> LEARN_RATE;
        } else if (name == "thread_num") {
            ifs >> THREAD_NUM;
        }
    }
}

void BonanzaMethodTrainer::train() {
    std::cout << "start BonanzaMethod" << std::endl;

    //学習開始時間の設定
    start_time_ = std::chrono::steady_clock::now();

    //評価関数ロード
    learning_model_.load("cnn.model");

    //棋譜を読み込む
    std::cout << "start loadGames ...";
    games_ = loadGames(KIFU_PATH, game_num_);
    std::cout << " done" << std::endl;
    std::cout << "games.size() = " << games_.size() << std::endl;

    //棋譜シャッフル
    std::default_random_engine engine(0);

    //学習情報の初期化
    learned_games_num = 0;
    learned_position_num = 0;
    succeeded_position_num = 0;
    game_index_ = 0;
    ordering_num_.assign(MAX_MOVE_LIST_SIZE, 0);

    //log_file_の設定
    log_file_.open("bonanza_method_log.txt", std::ios::out);
    log_file_ << "elapsed\tgame_num\tposition_num\tloss_average\ttop1\ttop3\ttop5\ttop10\ttop20\ttop50" << std::endl;
    log_file_ << std::fixed;

    THREAD_NUM = std::min(std::max(1u, THREAD_NUM), std::thread::hardware_concurrency());
    std::cout << "使用するスレッド数 : " << THREAD_NUM << std::endl;
    std::cout << std::fixed;

    struct TT {
        uint32_t label;
        float value;
    };
    std::vector<std::pair<std::string, TT>> data_buffer;
    for (const auto& game : games_) {
        Position pos;
        for (const auto& move : game.moves) {
            TT teacher;
            teacher.label = (uint32_t)move.toLabel();
            teacher.value = (float)(pos.color() == BLACK ? game.result : -game.result);
            data_buffer.emplace_back(pos.toSFEN(), teacher);
            pos.doMove(move);
        }
    }

    Position pos;

    O::MomentumSGD optimizer(LEARN_RATE);
    optimizer.add(learning_model_);

    Graph g;
    Graph::set_default(g);

    //学習開始
    for (int32_t epoch = 1; epoch <= 100; epoch++) {
        //データをシャッフル
        std::shuffle(data_buffer.begin(), data_buffer.end(), engine);

        for (int32_t step = 0; (step + 1) * BATCH_SIZE < data_buffer.size(); step++) {
            std::vector<float> input, value_teachers;
            std::vector<uint32_t> labels;
            for (int32_t b = 0; b < BATCH_SIZE; b++) {
                const auto& datum = data_buffer[step * BATCH_SIZE + b];
                pos.loadSFEN(datum.first);
                const auto& feature = pos.makeFeature();
                for (const auto& e : feature) {
                    input.push_back(e);
                }
                labels.push_back(datum.second.label);
                value_teachers.push_back(datum.second.value);
            }
            g.clear();
            auto loss = learning_model_.loss(input, labels, value_teachers, (uint32_t)BATCH_SIZE);
            timestamp();
            print(epoch);
            print(step);
            print(loss.first.to_float());
            print(loss.second.to_float());
            std::cout << std::endl;
            log_file_ << std::endl;
            optimizer.reset_gradients();
            loss.first.backward();
            loss.second.backward();
            optimizer.update();
        }

        learning_model_.save("cnn" + std::to_string(epoch) + ".model");
    }

    log_file_.close();
    std::cout << "finish BonanzaMethod" << std::endl;
}

void BonanzaMethodTrainer::testTrain() {
    std::cout << "start testTrain" << std::endl;

    //学習開始時間の設定
    start_time_ = std::chrono::steady_clock::now();

    //評価関数ロード
    learning_model_.load("cnn.model");

    game_num_ = 1;

    //棋譜を読み込む
    games_ = loadGames(KIFU_PATH, game_num_);
    std::cout << "games.size() = " << games_.size() << std::endl;

    struct TT {
        uint32_t label;
        float value;
    };
    std::vector<std::pair<std::string, TT>> data_buffer;
    for (const auto& game : games_) {
        Position pos;
        for (const auto& move : game.moves) {
            TT teacher;
            teacher.label = (uint32_t)move.toLabel();
            teacher.value = (float)(pos.color() == BLACK ? game.result : -game.result);
            data_buffer.emplace_back(pos.toSFEN(), teacher);
            pos.doMove(move);
        }
    }

    BATCH_SIZE = data_buffer.size();

    Position pos;

    O::MomentumSGD optimizer(0.01);
    optimizer.add(learning_model_);

    Graph g;
    Graph::set_default(g);

    //学習開始
    for (int32_t step = 1; step <= 500; step++) {
        std::vector<float> input, value_teachers;
        std::vector<uint32_t> labels;
        for (int32_t b = 0; b < BATCH_SIZE; b++) {
            const auto &datum = data_buffer[b];
            pos.loadSFEN(datum.first);
            const auto &feature = pos.makeFeature();
            for (const auto &e : feature) {
                input.push_back(e);
            }
            labels.push_back(datum.second.label);
            value_teachers.push_back(datum.second.value);
        }
        g.clear();
        auto loss = learning_model_.loss(input, labels, value_teachers, (uint32_t)BATCH_SIZE);
        print(step);
        print(loss.first.to_float());
        print(loss.second.to_float());
        auto l = loss.first + loss.second;
        std::cout << std::endl;
        optimizer.reset_gradients();
//            loss.first.backward();
//            loss.second.backward();
        l.backward();
        optimizer.update();
    }

    learning_model_.save(MODEL_PATH);
    nn->load(MODEL_PATH);
    pos.init();
    for (const auto& move : games_.front().moves) {
        pos.print();
        pos.doMove(move);
    }

    std::cout << "finish testTrain" << std::endl;
}