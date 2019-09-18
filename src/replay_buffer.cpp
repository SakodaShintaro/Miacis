﻿#include"replay_buffer.hpp"
#include<thread>
#include<iomanip>

const std::string ReplayBuffer::save_dir = "./learn_kifu/";

std::vector<LearningData> ReplayBuffer::makeBatch(int64_t batch_size) {
    //ロックの確保
    mutex_.lock();

    while (first_wait_ > 0) {
        mutex_.unlock();
        std::cout << "wait_remain = " << first_wait_ << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(first_wait_ / 100 + 10));
        mutex_.lock();
    }
    
    //現時点のpriorityの和を取得し,そこまでの範囲の一様分布生成器を作る
    float sum = segment_tree_.getSum();
    static std::mt19937 engine(0);
    std::uniform_real_distribution<float> dist(0.0, sum);

    //データを取得
    std::vector<LearningData> data;
    pre_indices_.clear();
    for (int32_t i = 0; i < batch_size; i++) {
        //データの取り出し及びインデックスを保存
        uint64_t index = segment_tree_.getIndex(dist(engine));
        data.push_back(data_[index]);
        pre_indices_.push_back(index);
    }

    //ロックの解放
    mutex_.unlock();
    return data;
}

void ReplayBuffer::push(Game &game) {
    mutex_.lock();

    Position pos;

    static int64_t num = 0;
    if (++num % output_interval_ == 0) {
        game.writeKifuFile(save_dir);
    }

    //まずは最終局面まで動かす
    for (const auto& e : game.elements) {
        pos.doMove(e.move);
    }

    assert(Game::RESULT_WHITE_WIN <= game.result && game.result <= Game::RESULT_BLACK_WIN);

    //先手から見た勝率,分布.指数移動平均で動かしていく.最初は結果によって初期化
    double win_rate_for_black = game.result;

    for (int32_t i = (int32_t)game.elements.size() - 1; i >= 0; i--) {
        //i番目の指し手を教師とするのは1手戻した局面
        pos.undo();

        auto& e = game.elements[i];

        //探索結果を先手から見た値に変換
        double curr_win_rate = (pos.color() == BLACK ? e.score : MAX_SCORE + MIN_SCORE - e.score);
        //混合
        win_rate_for_black = lambda_ * win_rate_for_black + (1.0 - lambda_) * curr_win_rate;

        double teacher_signal = (pos.color() == BLACK ? win_rate_for_black : MAX_SCORE + MIN_SCORE - win_rate_for_black);

#ifdef USE_CATEGORICAL
        //teacherにコピーする
        e.value_teacher = valueToIndex(teacher_signal);
#else
        //teacherにコピーする
        e.value_teacher = (FloatType) (teacher_signal);
#endif

        //このデータを入れる位置を取得
        int64_t change_index = segment_tree_.getIndexToPush();

        //そこのデータを入れ替える
        data_[change_index].move = e.move;
        data_[change_index].SFEN = pos.toSFEN();
        //data_[change_index].policy = e.policy_teacher;
        data_[change_index].value = e.value_teacher;

        //priorityを計算
        float priority = 0.0f;

        //Policy損失
        priority += -std::log(e.nn_output_policy[e.move.toLabel()] + 1e-9f);
//        for (const auto& p : e.policy_teacher) {
//            priority += -p.second * std::log(e.nn_output_policy[p.first] + 1e-9f);
//        }

        //Value損失
#ifdef USE_CATEGORICAL
        //float priority = -2 * std::log(e.nn_output_policy[e.move.toLabel()] + 1e-10f) - std::log(e.nn_output_value[e.teacher.value] + 1e-10f);
        priority += -std::log(e.nn_output_value[e.value_teacher] + 1e-9f);
#else
        //float priority = -2 * std::log(e.nn_output_policy[e.move.toLabel()] + 1e-10f) + std::pow(e.nn_output_value - e.teacher.value, 2.0f);
        priority += std::pow(e.nn_output_value - e.value_teacher, 2.0f);
#endif
        //segment_treeのpriorityを更新
        segment_tree_.update(change_index, std::pow(priority * 2, alpha_));

        if (first_wait_ > 0) {
            first_wait_--;
        }
    }

    mutex_.unlock();
}

void ReplayBuffer::update(const std::vector<float>& loss) {
    mutex_.lock();

    assert(loss.size() == pre_indices_.size());
    for (uint64_t i = 0; i < loss.size(); i++) {
        segment_tree_.update(pre_indices_[i], std::pow(loss[i], alpha_));
    }

    mutex_.unlock();
}