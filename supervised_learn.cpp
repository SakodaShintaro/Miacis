#include"learn.hpp"
#include"game.hpp"
#include"operate_params.hpp"
#include<fstream>
#include<iostream>
#include<cassert>

void supervisedLearn() {
    //学習開始時間の設定
    auto start_time_ = std::chrono::steady_clock::now();

    //設定ファイルの読み込み
    std::ifstream ifs("supervised_learn_settings.txt");
    if (!ifs) {
        std::cerr << "fail to open supervised_learn_settings.txt" << std::endl;
        assert(false);
    }

    //後で読み込みの成功を確認するために不適当な値で初期化
    std::string kifu_path;
    int64_t game_num = -1;
    int64_t batch_size = -1;
    float learn_rate = -1;
    float momentum = -1;
    float weight_decay = -1;
    int64_t patience_limit = -1;

    std::string name;
    while (ifs >> name) {
        if (name == "kifu_path") {
            ifs >> kifu_path;
        } else if (name == "game_num") {
            ifs >> game_num;
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
    assert(game_num > 0);
    assert(batch_size > 0);
    assert(learn_rate >= 0);
    assert(momentum >= 0);
    assert(weight_decay >= 0);
    assert(patience_limit > 0);

    //評価関数読み込み
#ifdef USE_LIBTORCH
    NeuralNetwork learning_model_;
    torch::load(learning_model_, MODEL_PATH);
    torch::save(learning_model_, MODEL_PREFIX + "_before_learn.model");
#else
    NeuralNetwork<Node> learning_model_;
    learning_model_.load(MODEL_PATH);
    learning_model_.save(MODEL_PREFIX + "_before_learn.model");
#endif

    //棋譜を読み込む
    std::cout << "start loadGames ..." << std::flush;
    std::vector<Game> games = loadGames(kifu_path, game_num);
    std::cout << " done.  games.size() = " << games.size() << std::endl;

    //学習データを局面単位にバラして保持
    std::vector<std::pair<std::string, TeacherType>> data_buffer;
    for (const auto& game : games) {
        Position pos;
        for (const auto& move : game.moves) {
            TeacherType teacher;
            teacher.policy = move.toLabel();
#ifdef USE_CATEGORICAL
            teacher.value = valueToIndex(pos.color() == BLACK ? game.result : MAX_SCORE + MIN_SCORE - game.result);
#else
            teacher.value = (float)(pos.color() == BLACK ? game.result : MAX_SCORE + MIN_SCORE - game.result);
#endif
            data_buffer.emplace_back(pos.toSFEN(), teacher);
            pos.doMove(move);
        }
    }

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
    std::cout << "learn_data_size = " << data_buffer.size() << ", validation_data_size" << validation_size << std::endl;

    //validation用の変数宣言
    float min_loss = 1e10;
    int32_t patience = 0;

    //学習推移のログファイル
    std::ofstream learn_log("supervised_learn_log.txt", std::ios::out);
    learn_log << "time\tepoch\tstep\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;
    std::cout << "time\tepoch\tstep\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;

    //validation結果のログファイル
    std::ofstream validation_log("validation_log.txt");
    validation_log << "epoch\ttime\tsum_loss\tpolicy_loss\tvalue_loss" << std::fixed << std::endl;

    //optimizerの設定
#ifdef USE_LIBTORCH
    torch::optim::SGDOptions sgd_option(learn_rate);
    sgd_option.momentum(momentum);
    sgd_option.weight_decay(weight_decay);
    torch::optim::SGD optimizer(learning_model_->parameters(), sgd_option);
#else
    O::MomentumSGD optimizer(learn_rate);
    optimizer.add(learning_model_);
    optimizer.set_weight_decay(weight_decay);

    //グラフの設定
    Graph g;
    Graph::set_default(g);
#endif

    //学習開始
    for (int32_t epoch = 1; epoch <= 1000; epoch++) {
        //データをシャッフル
        std::shuffle(data_buffer.begin(), data_buffer.end(), engine);

        for (int32_t step = 0; (step + 1) * batch_size <= data_buffer.size(); step++) {
            std::vector<float> inputs;
            std::vector<PolicyTeacherType> policy_teachers;
            std::vector<ValueTeacherType> value_teachers;
            std::tie(inputs, policy_teachers, value_teachers) = getBatch(data_buffer, step * batch_size, batch_size);

#ifdef USE_LIBTORCH
            optimizer.zero_grad();
            auto loss = learning_model_->loss(inputs, policy_teachers, value_teachers);
            if (step % 100 == 0) {
                auto p_loss = loss.first.item<float>();
                auto v_loss = loss.second.item<float>();
                std::cout << elapsedTime(start_time_)  << "\t" << epoch << "\t" << step << "\t" << p_loss << "\t" << v_loss << std::endl;
                learn_log << elapsedHours(start_time_) << "\t" << epoch << "\t" << step << "\t" << p_loss << "\t" << v_loss << std::endl;
            }
            auto sum_loss = loss.first + loss.second;
            sum_loss.backward();
            optimizer.step();
#else
            g.clear();
            auto loss = learning_model_.loss(inputs, policy_teachers, value_teachers);
            if (step % 100 == 0) {
                float p_loss = loss.first.to_float();
                float v_loss = loss.second.to_float();
                std::cout << elapsedTime(start_time_)  << "\t" << epoch << "\t" << step << "\t" << p_loss << "\t" << v_loss << std::endl;
                learn_log << elapsedHours(start_time_) << "\t" << epoch << "\t" << step << "\t" << p_loss << "\t" << v_loss << std::endl;
            }
            optimizer.reset_gradients();
            loss.first.backward();
            loss.second.backward();
            optimizer.update();
#endif
        }

        //学習中のパラメータを書き出して推論用のモデルで読み込む
#ifdef USE_LIBTORCH
        torch::save(learning_model_, MODEL_PATH);
        torch::load(nn, MODEL_PATH);
#else
        learning_model_.save(MODEL_PATH);
        nn->load(MODEL_PATH);
#endif
        int32_t num = 0;
        float policy_loss = 0.0, value_loss = 0.0;
        for (int32_t i = 0; (i + 1) * batch_size <= validation_size; i++, num++) {
            std::vector<float> inputs;
            std::vector<PolicyTeacherType> policy_teachers;
            std::vector<ValueTeacherType> value_teachers;
            std::tie(inputs, policy_teachers, value_teachers) = getBatch(validation_data, i * batch_size, batch_size);
            auto loss = nn->loss(inputs, policy_teachers, value_teachers);
#ifdef USE_LIBTORCH
            policy_loss += loss.first.item<float>();
            value_loss += loss.second.item<float>();
#else
            policy_loss += loss.first.to_float();
            value_loss += loss.second.to_float();
#endif
        }
        policy_loss /= num;
        value_loss /= num;
        float sum_loss = policy_loss + value_loss;

        std::cout      << epoch << "\t" << elapsedTime(start_time_)  << "\t" << sum_loss << "\t" << policy_loss << "\t" << value_loss << std::endl;
        validation_log << epoch << "\t" << elapsedHours(start_time_) << "\t" << sum_loss << "\t" << policy_loss << "\t" << value_loss << std::endl;

        if (sum_loss < min_loss) {
            min_loss = sum_loss;
            patience = 0;
#ifdef USE_LIBTORCH
            torch::save(learning_model_, MODEL_PREFIX + "_supervised_best.model");
#else
            learning_model_.save(MODEL_PREFIX + "_supervised_best.model");
#endif
        } else if (++patience >= patience_limit) {
#ifdef USE_LIBTORCH
            optimizer.options.learning_rate_ /= 2;
#else
            optimizer.set_learning_rate_scaling(optimizer.get_learning_rate_scaling() / 2);
#endif
            break;
        }
    }

    std::cout << "finish SupervisedLearn" << std::endl;
}