#include"replay_buffer.hpp"
#include"operate_params.hpp"
#include<thread>
#include<iomanip>

void ReplayBuffer::makeBatch(int64_t batch_size, std::vector<float>& inputs,
                             std::vector<PolicyTeacherType>& policy_teachers,
                             std::vector<ValueTeacherType>& value_teachers) {
    //ロックの確保
    mutex_.lock();

    //現時点のpriorityの和を取得し,そこまでの範囲の一様分布生成器を作る
    float sum = segment_tree_.getSum();
    std::mt19937 engine(0);
    std::uniform_real_distribution<float> dist(0.0, sum);

    //データを入れる
    Position pos;
    inputs.clear();
    policy_teachers.clear();
    value_teachers.clear();
    for (int32_t i = 0; i < batch_size; i++) {
        //データの取り出し
        std::string sfen;
        TeacherType teacher;
        std::tie(sfen, teacher) = data_[segment_tree_.getIndex(dist(engine))];

        //入力特徴量の確保
        pos.loadSFEN(sfen);
        auto feature = pos.makeFeature();
        inputs.insert(inputs.end(), feature.begin(), feature.end());

        //policyの教師
        policy_teachers.push_back(teacher.policy);

        //valueの教師
#ifdef USE_CATEGORICAL
        value_teachers.push_back(valueToIndex(teacher.value));
#else
        value_teachers.push_back(teacher.value);
#endif
    }

    //ロックの解放
    mutex_.unlock();
}

void ReplayBuffer::push(Game &game) {
    mutex_.lock();

    Position pos;

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
        double curr_win_rate = (pos.color() == BLACK ? e.move.score : MAX_SCORE + MIN_SCORE - e.move.score);
        //混合
        win_rate_for_black = lambda_ * win_rate_for_black + (1.0 - lambda_) * curr_win_rate;

        double teacher_signal = (pos.color() == BLACK ? win_rate_for_black : MAX_SCORE + MIN_SCORE - win_rate_for_black);

#ifdef USE_CATEGORICAL
        //teacherにコピーする
        e.teacher.value = valueToIndex(teacher_signal);
#else
        //teacherにコピーする
        e.teacher.value = (CalcType) (teacher_signal);
#endif

        //priorityが最小のものを取る
        auto t = priority_queue_.top();
        priority_queue_.pop();

        //そこのデータを入れ替える
        data_[t.index] = std::make_tuple(pos.toSFEN(), e.teacher);

        //priorityはmoveのscoreに入れられている.そこに時間ボーナスを入れる
#ifdef USE_CATEGORICAL
        float priority = 0.0;
        assert(false);
#else
        float priority = -std::log(e.nn_output_policy[e.move.toLabel()] + 1e-10f) + std::pow(e.nn_output_value - e.teacher.value, 2.0f) + priority_time_bonus_;
#endif
        std::cout << e.nn_output_policy[e.move.toLabel()] << ", " << e.nn_output_value<< ", " <<  e.teacher.value << std::endl;
        std::cout << "priority = " << priority << std::endl;

        //pqとsegment_treeのpriorityを更新
        priority_queue_.push(Element(priority, t.index));
        segment_tree_.update(t.index, priority);
    }

    priority_time_bonus_ += 1e-4;

    mutex_.unlock();
}

void ReplayBuffer::clear() {
    mutex_.lock();
    data_.clear();
    data_.shrink_to_fit();
    mutex_.unlock();
}