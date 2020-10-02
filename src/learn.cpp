﻿#include "learn.hpp"
#include "game.hpp"
#include "hyperparameter_loader.hpp"
#include "include_switch.hpp"
#include <iomanip>
#include <random>
#include <sstream>

std::string elapsedTime(const std::chrono::steady_clock::time_point& start) {
    auto elapsed = std::chrono::steady_clock::now() - start;
    int64_t seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();

    std::stringstream ss;

    //hh:mm:ssで文字列化
    int64_t minutes = seconds / 60;
    seconds %= 60;
    int64_t hours = minutes / 60;
    minutes %= 60;
    ss << std::setfill('0') << std::setw(3) << hours << ":" << std::setfill('0') << std::setw(2) << minutes << ":"
       << std::setfill('0') << std::setw(2) << seconds;
    return ss.str();
}

float elapsedHours(const std::chrono::steady_clock::time_point& start) {
    auto elapsed = std::chrono::steady_clock::now() - start;
    int64_t seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    return seconds / 3600.0;
}

std::array<float, LOSS_TYPE_NUM> validation(NeuralNetwork nn, const std::vector<LearningData>& validation_data,
                                            uint64_t batch_size) {
    uint64_t index = 0;
    std::array<float, LOSS_TYPE_NUM> losses{};
    torch::NoGradGuard no_grad_guard;
    Position pos;
    while (index < validation_data.size()) {
        //バッチサイズ分データを確保
        std::vector<LearningData> curr_data;
        while (index < validation_data.size() && curr_data.size() < batch_size) {
            curr_data.push_back(validation_data[index++]);
        }

        //計算
        std::array<torch::Tensor, LOSS_TYPE_NUM> loss = nn->loss(curr_data);

        for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
            if (i == VALUE_LOSS_INDEX) {
                //valueだけは別計算する
                continue;
            }
            losses[i] += loss[i].sum().item<float>();
        }

#ifdef USE_CATEGORICAL
        //categoricalモデルのときは冗長だがもう一度順伝播を行って損失を手動で計算
        std::vector<float> inputs;
        for (const LearningData& datum : curr_data) {
            pos.fromStr(datum.position_str);
            std::vector<float> feature = pos.makeFeature();
            inputs.insert(inputs.end(), feature.begin(), feature.end());
        }
        std::pair<std::vector<PolicyType>, std::vector<ValueType>> y = nn->policyAndValueBatch(inputs);
        const std::vector<ValueType>& values = y.second;
        for (uint64_t i = 0; i < values.size(); i++) {
            float e = expOfValueDist(values[i]);
            float vt = (curr_data[i].value == BIN_SIZE - 1 ? MAX_SCORE : MIN_SCORE);
            losses[VALUE_LOSS_INDEX] += (e - vt) * (e - vt);
        }
#else
        //scalarモデルのときはそのまま損失を加える
        losses[VALUE_LOSS_INDEX] += loss[VALUE_LOSS_INDEX].sum().item<float>();
#endif
    }

    //データサイズで割って平均
    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
        losses[i] /= validation_data.size();
    }

    return losses;
}

std::vector<LearningData> loadData(const std::string& file_path, bool data_augmentation, float rate_threshold) {
    //棋譜を読み込めるだけ読み込む
    std::vector<Game> games = loadGames(file_path, rate_threshold);

    //データを局面単位にバラす
    std::vector<LearningData> data;
    for (const Game& game : games) {
        Position pos;
        for (uint64_t i = 0; i + LEARNING_RANGE < game.elements.size(); i++) {
            LearningData datum;
            datum.position_str = pos.toStr();

            for (uint64_t j = 0; j < LEARNING_RANGE; j++) {
                datum.moves[j] = game.elements[i + j].move;
#ifdef USE_CATEGORICAL
                datum.value[j] =
                    valueToIndex(((pos.color() + j) % 2 == BLACK ? game.result : MAX_SCORE + MIN_SCORE - game.result));
#else
                datum.value[j] = (float)((pos.color() + j) % 2 == BLACK ? game.result : MAX_SCORE + MIN_SCORE - game.result);
#endif
            }
            data.push_back(datum);
            pos.doMove(game.elements[i].move);
        }
    }

    return data;
}

void initParams() {
    NeuralNetwork nn;
    torch::save(nn, NeuralNetworkImpl::DEFAULT_MODEL_NAME);
    std::cout << "初期化したパラメータを" << NeuralNetworkImpl::DEFAULT_MODEL_NAME << "に出力" << std::endl;
}

std::vector<std::string> childFiles(const std::string& file_path) {
#ifdef _MSC_VER
    namespace sys = std::filesystem;
#elif __GNUC__
    namespace sys = std::experimental::filesystem;
#endif

    const sys::path dir(file_path);
    std::vector<std::string> child_files;
    for (sys::directory_iterator p(dir); p != sys::directory_iterator(); p++) {
        child_files.push_back(p->path().string());
    }
    return child_files;
}