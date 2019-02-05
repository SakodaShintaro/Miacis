#include"replay_buffer.hpp"
#include"operate_params.hpp"

#include<thread>

std::tuple<std::vector<float>, std::vector<uint32_t>, std::vector<ValueTeacher>> ReplayBuffer::makeBatch(int32_t batch_size) {
    //ロックの確保
    mutex_.lock();

    //一番最初だけ十分量に達するまで待つ
    while (data_.size() < first_wait) {
        mutex_.unlock();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        mutex_.lock();
    }

    //とりあえずランダムに取得
    static std::mt19937 engine(0);
    std::uniform_int_distribution<unsigned long> dist(0, data_.size() - 1);

    Position pos;

    std::vector<float> inputs;
    std::vector<uint32_t> policy_labels;
    std::vector<ValueTeacher> value_teachers;
    for (int32_t i = 0; i < batch_size; i++) {
        //データの取り出し
        std::string sfen;
        uint32_t policy_label;
        ValueTeacher value;
        std::tie(sfen, policy_label, value) = data_[dist(engine)];

        //入力特徴量の確保
        pos.loadSFEN(sfen);
        auto feature = pos.makeFeature();
        inputs.resize(inputs.size() + feature.size());
        std::copy(feature.begin(), feature.end(), inputs.end() - feature.size());

        //policyの教師
        policy_labels.push_back(policy_label);

        //valueの教師
#ifdef USE_CATEGORICAL
        value_teachers.push_back(valueToIndex(value));
#else
        value_teachers.push_back(value);
#endif
    }

    //ロックの解放
    mutex_.unlock();

    return std::make_tuple(inputs, policy_labels, value_teachers);
}

void ReplayBuffer::push(Game &game) {
    mutex_.lock();

    Position pos;

    //まずは最終局面まで動かす
    for (auto move : game.moves) {
        pos.doMove(move);
    }

    assert(Game::RESULT_WHITE_WIN <= game.result && game.result <= Game::RESULT_BLACK_WIN);

    //先手から見た勝率,分布.指数移動平均で動かしていく.最初は結果によって初期化
    double win_rate_for_black = game.result;

    for (int32_t i = (int32_t) game.moves.size() - 1; i >= 0; i--) {
        //i番目の指し手を教師とするのは1手戻した局面
        pos.undo();

        //探索結果を先手から見た値に変換
        double curr_win_rate = (pos.color() == BLACK ? game.moves[i].score : MAX_SCORE + MIN_SCORE - game.moves[i].score);
        //混合
        win_rate_for_black = lambda * win_rate_for_black + (1.0 - lambda) * curr_win_rate;

        double teacher_signal = (pos.color() == BLACK ? win_rate_for_black : MAX_SCORE + MIN_SCORE - win_rate_for_black);

#ifdef USE_CATEGORICAL
        //teacherにコピーする
        game.teachers[i].value = valueToIndex(teacher_signal);
#else
        //teacherにコピーする
        game.teachers[i].value = (CalcType) (teacher_signal);
#endif
        //スタックに詰める
        data_.emplace_back(pos.toSFEN(), game.teachers[i].policy, game.teachers[i].value);
    }
    mutex_.unlock();
}