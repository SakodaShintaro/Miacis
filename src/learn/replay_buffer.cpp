﻿#include "replay_buffer.hpp"
#include "../common.hpp"
#include "../shogi/position.hpp"
#include <iomanip>
#include <random>
#include <thread>

const std::string ReplayBuffer::KIFU_SAVE_DIR = "./learn_kifu/";

std::vector<LearningData> ReplayBuffer::makeBatch(int64_t batch_size) {
    //ロックの確保
    mutex_.lock();

    while (first_wait_ > 0) {
        mutex_.unlock();
        std::cout << "\rwait_remain = " << first_wait_ << "  " << std::flush;
        std::this_thread::sleep_for(std::chrono::seconds(10));
        mutex_.lock();
    }

    //現時点のpriorityの和を取得し,そこまでの範囲の一様分布生成器を作る
    float sum = segment_tree_.getSum();
    std::uniform_real_distribution<float> dist(0.0, sum);

    //データを取得
    std::vector<LearningData> data;
    pre_indices_.clear();
    for (int64_t i = 0; i < batch_size; i++) {
        //データの取り出し及びインデックスを保存
        uint64_t index = segment_tree_.getIndex(dist(engine));
        data.push_back(data_[index]);
        pre_indices_.push_back(index);
    }

    //ロックの解放
    mutex_.unlock();
    return data;
}

void ReplayBuffer::push(Game& game) {
    mutex_.lock();

    Position pos;

    static int64_t total_kifu_num = 0;
    if (++total_kifu_num % output_interval_ == 0) {
        game.writeKifuFile(KIFU_SAVE_DIR);
    }

    //まずは最終局面まで動かす
    for (const OneTurnElement& e : game.elements) {
        pos.doMove(e.move);
    }

    if (game.result < MIN_SCORE || MAX_SCORE < game.result) {
        std::cout << "game.result < MIN_SCORE || MAX_SCORE < game.result" << std::endl;
        std::exit(1);
    }

    //先手から見た勝率,分布.指数移動平均で動かしていく.最初は結果によって初期化
    float win_rate_for_black = game.result;

    for (int32_t i = (int32_t)game.elements.size() - 1; i >= 0; i--) {
        //i番目の指し手を教師とするのは1手戻した局面
        pos.undo();

        OneTurnElement& e = game.elements[i];

        //探索結果を先手から見た値に変換
        float curr_win_rate = (pos.color() == BLACK ? e.score : MAX_SCORE + MIN_SCORE - e.score);
        //混合
        win_rate_for_black = lambda_ * win_rate_for_black + (1.0 - lambda_) * curr_win_rate;

        float teacher_signal = (pos.color() == BLACK ? win_rate_for_black : MAX_SCORE + MIN_SCORE - win_rate_for_black);

        ValueTeacherType value_teacher = teacher_signal;

        //priorityを計算
        float priority = 0.0f;

        //Policy損失
        for (const std::pair<int32_t, float>& p : e.policy_teacher) {
            priority += -p.second * std::log(e.nn_output_policy[p.first] + 1e-9f);
        }

        //Value損失
#ifdef USE_CATEGORICAL
        const float nn_output_value = expOfValueDist(e.nn_output_value);
        priority += std::pow(nn_output_value - value_teacher, 2.0f);
#else
#ifdef USE_SIGMOID
        constexpr float eps = 1e-5f;
        priority +=
            -value_teacher * std::log(e.nn_output_value + eps) - (1.0 - value_teacher) * std::log(1 - e.nn_output_value + eps);
#else
        priority += std::pow(e.nn_output_value - value_teacher, 2.0f);
#endif
#endif

        for (int64_t j = 0; j < (data_augmentation_ ? Position::DATA_AUGMENTATION_PATTERN_NUM : 1); j++) {
            //方策の教師を適切に変換
            PolicyTeacherType policy_teacher = e.policy_teacher;
            for (std::pair<int32_t, float>& element : policy_teacher) {
                element.first = Move::augmentLabel(element.first, j);
            }

            //このデータを入れる位置を取得
            int64_t change_index = segment_tree_.getIndexToPush();

            //そこのデータを入れ替える
            data_[change_index].position_str = Position::augmentStr(pos.toStr(), j);
            data_[change_index].policy = policy_teacher;
            data_[change_index].value = value_teacher;

            //segment_treeのpriorityを更新
            segment_tree_.update(change_index, std::pow(priority * 2, alpha_));

            //データを加えた数をカウント
            //拡張したデータは一つとして数えた方が良いかもしれない？
            total_num_++;
            if (first_wait_ > 0) {
                first_wait_--;
            }
        }
    }

    mutex_.unlock();
}

void ReplayBuffer::update(const std::vector<float>& loss) {
    mutex_.lock();

    if (loss.size() != pre_indices_.size()) {
        std::cout << "loss.size() != pre_indices_.size()" << std::endl;
        std::exit(1);
    }
    for (uint64_t i = 0; i < loss.size(); i++) {
        segment_tree_.update(pre_indices_[i], std::pow(loss[i], alpha_));
    }

    mutex_.unlock();
}

void ReplayBuffer::fillByKifu(const std::string& file_path, float rate_threshold) {
    std::vector<LearningData> data = loadData(file_path, data_augmentation_, rate_threshold);
    std::cout << "data.size() = " << data.size() << ", max_size_ = " << max_size_ << std::endl;
    for (int64_t i = 0; i < std::min(max_size_, (int64_t)data.size()); i++) {
        //このデータを入れる位置を取得
        int64_t change_index = segment_tree_.getIndexToPush();

        //そこのデータを入れ替える
        data_[change_index] = data[i];

        //segment_treeのpriorityを更新
        segment_tree_.update(change_index, 3.0f);

        //データを加えた数をカウント
        //拡張したデータは一つとして数えた方が良いかもしれない？
        total_num_++;
        if (first_wait_ > 0) {
            first_wait_--;
        }
    }
}