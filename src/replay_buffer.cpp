#include"replay_buffer.hpp"
#include<thread>
#include<iomanip>

const std::string ReplayBuffer::save_dir = "./learn_kifu";

void ReplayBuffer::makeBatch(int64_t batch_size, std::vector<float>& inputs,
                             std::vector<PolicyTeacherType>& policy_teachers,
                             std::vector<ValueTeacherType>& value_teachers) {
    //ロックの確保
    mutex_.lock();

    //一番最初だけ十分量に達するまで待つ
    while (data_.size() < first_wait_) {
        double percentage = 100.0 * data_.size() / first_wait_;
        std::cout << "replay_buffer.size() = " << data_.size() << " (" << percentage << "%)" << std::endl;
        mutex_.unlock();
        std::this_thread::sleep_for(std::chrono::seconds((uint64_t)((100 - percentage) * 2 + 10)));
        mutex_.lock();
    }

    //最大量を超えていたらその分を削除
    if (data_.size() > max_size_) {
        data_.erase(data_.begin(), data_.begin() + data_.size() - max_size_);
        data_.shrink_to_fit();
    }

    //とりあえずランダムに取得
    static std::mt19937 engine(0);
    std::uniform_int_distribution<uint64_t> dist(0, data_.size() - 1);

    Position pos;

    assert(inputs.size() == batch_size * INPUT_CHANNEL_NUM * SQUARE_NUM);
    assert(policy_teachers.size() == batch_size);
    assert(value_teachers.size() == batch_size);
    for (int32_t i = 0; i < batch_size; i++) {
        //データの取り出し
        std::string sfen;
        PolicyTeacherType policy;
        ValueTeacherType value;
        std::tie(sfen, policy, value) = data_[dist(engine)];

        //入力特徴量の確保
        pos.loadSFEN(sfen);
        auto feature = pos.makeFeature();
        std::copy(feature.begin(), feature.end(), inputs.begin() + i * INPUT_CHANNEL_NUM * SQUARE_NUM);

        //教師データ
        policy_teachers[i] = policy;
        value_teachers[i] = value;
    }

    //ロックの解放
    mutex_.unlock();
}

void ReplayBuffer::push(Game &game) {
    mutex_.lock();

    Position pos;

    static int64_t num = 0;
    if (++num % 500 == 0) {
        game.writeKifuFile(save_dir);
    }

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
        win_rate_for_black = lambda_ * win_rate_for_black + (1.0 - lambda_) * curr_win_rate;

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