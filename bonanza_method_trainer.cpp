#include"bonanza_method_trainer.hpp"
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
        } else if (name == "learn_rate") {
            ifs >> LEARN_RATE;
        } else if (name == "L2_regularization_coeff") {
            ifs >> L2_REGULARIZATION_COEFF;
        }
    }
}

void BonanzaMethodTrainer::train() {
    std::cout << "start BonanzaMethod" << std::endl;

    //学習開始時間の設定
    start_time_ = std::chrono::steady_clock::now();

    //評価関数ロード
    learning_model_.load(MODEL_PATH);
    learning_model_.save("before_bonanza_method.model");

    //棋譜を読み込む
    std::cout << "start loadGames ..." << std::flush;
    games_ = loadGames(KIFU_PATH, game_num_);
    std::cout << " done" << std::endl;
    std::cout << "games.size() = " << games_.size() << std::endl;

    //棋譜シャッフル
    std::default_random_engine engine(0);

    //log_file_の設定
    log_file_.open("bonanza_method_log.txt", std::ios::out);
    log_file_ << "elapsed\tepoch\tstep\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;
    std::cout << std::fixed;

    //validation結果のログファイル
    std::ofstream validation_log("validation_log.txt");
    validation_log << "time\tepoch\tloss" << std::fixed << std::endl;

    std::vector<std::pair<std::string, TeacherType>> data_buffer;
    auto getBatch = [this](const std::vector<std::pair<std::string, TeacherType>>& data_buf, int64_t index) {
        Position pos;
        std::vector<float> inputs;
        std::vector<uint32_t> policy_labels;
        std::vector<ValueTeacher> value_teachers;
        for (int32_t b = 0; b < BATCH_SIZE; b++) {
            const auto& datum = data_buf[index + b];
            pos.loadSFEN(datum.first);
            const auto& feature = pos.makeFeature();
            for (const auto& e : feature) {
                inputs.push_back(e);
            }
            policy_labels.push_back(datum.second.policy);
            value_teachers.push_back(datum.second.value);
        }
        return std::make_tuple(inputs, policy_labels, value_teachers);
    };

    for (const auto& game : games_) {
        Position pos;
        for (const auto& move : game.moves) {
            TeacherType teacher;
            teacher.policy = (uint32_t)move.toLabel();
            teacher.value = (float)(pos.color() == BLACK ? game.result : -game.result);
            data_buffer.emplace_back(pos.toSFEN(), teacher);
            pos.doMove(move);
        }
    }

    //データをシャッフル
    std::shuffle(data_buffer.begin(), data_buffer.end(), engine);

    //validationデータを確保
    std::vector<std::pair<std::string, TeacherType>> validation_data;
    auto validation_size = (int32_t)(data_buffer.size() * 0.1) / BATCH_SIZE * BATCH_SIZE;
    assert(validation_size != 0);
    for (int32_t i = 0; i < validation_size; i++) {
        validation_data.push_back(data_buffer.back());
        data_buffer.pop_back();
    }
    std::cout << data_buffer.size() << std::endl;

    Position pos;

    O::MomentumSGD optimizer(LEARN_RATE);
    optimizer.add(learning_model_);

    Graph g;
    Graph::set_default(g);

    //validation用
    float min_loss = 1e10;
    int32_t patience = 0;

    //学習開始
    for (int32_t epoch = 1; epoch <= 100; epoch++) {
        //データをシャッフル
        std::shuffle(data_buffer.begin(), data_buffer.end(), engine);

        for (int32_t step = 0; (step + 1) * BATCH_SIZE <= data_buffer.size(); step++) {
            std::vector<float> inputs;
            std::vector<uint32_t> policy_labels;
            std::vector<ValueTeacher> value_teachers;
            std::tie(inputs, policy_labels, value_teachers) = getBatch(data_buffer, step * BATCH_SIZE);
            g.clear();
            auto loss = learning_model_.loss(inputs, policy_labels, value_teachers);
            if (step % 100 == 0) {
                learning_model_.save(MODEL_PATH);
                timestamp();
                print(epoch);
                print(step);
                print(step * BATCH_SIZE);
                print(loss.first.to_float());
                print(loss.second.to_float());
                std::cout << std::endl;
                log_file_ << std::endl;
            }
            optimizer.reset_gradients();
            loss.first.backward();
            loss.second.backward();
            optimizer.update();
        }

        learning_model_.save(MODEL_PATH);

        //ここからvalidation
        nn->load(MODEL_PATH);

        int32_t num = 0;
        float curr_loss = 0.0;
        for (int32_t i = 0; (i + 1) * BATCH_SIZE <= validation_size; i++, num++) {
            std::vector<float> inputs;
            std::vector<uint32_t> policy_labels;
            std::vector<ValueTeacher> value_teachers;
            std::tie(inputs, policy_labels, value_teachers) = getBatch(validation_data, i * BATCH_SIZE);
            auto loss = nn->loss(inputs, policy_labels, value_teachers);
            curr_loss += (loss.first.to_float() + loss.second.to_float());
        }
        curr_loss /= num;

        validation_log << epoch << "\t" << elapsedHours() << "\t" << curr_loss << std::endl;

        if (curr_loss < min_loss) {
            min_loss = curr_loss;
            patience = 0;
            learning_model_.save("best.model");
        } else if (++patience >= 5) {
            break;
        }
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

    std::vector<std::pair<std::string, TeacherType>> data_buffer;
    for (const auto& game : games_) {
        Position pos;
        for (const auto& move : game.moves) {
            TeacherType teacher;
            teacher.policy = (uint32_t)move.toLabel();
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
        std::vector<float> input;
        std::vector<ValueTeacher> value_teachers;
        std::vector<uint32_t> labels;
        for (int32_t b = 0; b < BATCH_SIZE; b++) {
            const auto &datum = data_buffer[b];
            pos.loadSFEN(datum.first);
            const auto &feature = pos.makeFeature();
            for (const auto &e : feature) {
                input.push_back(e);
            }
            labels.push_back(datum.second.policy);
            value_teachers.push_back(datum.second.value);
        }
        g.clear();
        auto loss = learning_model_.loss(input, labels, value_teachers);
        print(step);
        print(loss.first.to_float());
        print(loss.second.to_float());
        std::cout << std::endl;
        optimizer.reset_gradients();
        loss.first.backward();
        loss.second.backward();
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