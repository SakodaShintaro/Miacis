#include"learn.hpp"
#include"game.hpp"
#include<sstream>
#include<iomanip>

std::string elapsedTime(const std::chrono::steady_clock::time_point& start) {
    auto elapsed = std::chrono::steady_clock::now() - start;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();

    std::stringstream ss;

    //hh:mm:ssで文字列化
    auto minutes = seconds / 60;
    seconds %= 60;
    auto hours = minutes / 60;
    minutes %= 60;
    ss << std::setfill('0') << std::setw(3) << hours << ":"
       << std::setfill('0') << std::setw(2) << minutes << ":"
       << std::setfill('0') << std::setw(2) << seconds;
    return ss.str();
}

double elapsedHours(const std::chrono::steady_clock::time_point& start) {
    auto elapsed = std::chrono::steady_clock::now() - start;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    return seconds / 3600.0;
}

std::tuple<std::vector<float>, std::vector<PolicyTeacherType>, std::vector<ValueTeacherType>>
getBatch(const std::vector<std::pair<std::string, TeacherType>>& data_buf, int64_t index, int64_t batch_size) {
    Position pos;
    std::vector<float> inputs;
    std::vector<PolicyTeacherType> policy_teachers;
    std::vector<ValueTeacherType> value_teachers;
    for (int32_t b = 0; b < batch_size; b++) {
        const auto& datum = data_buf[index + b];
        pos.loadSFEN(datum.first);
        const auto feature = pos.makeFeature();
        inputs.insert(inputs.end(), feature.begin(), feature.end());
        policy_teachers.push_back(datum.second.policy);
        value_teachers.push_back(datum.second.value);
    }
    return std::make_tuple(inputs, policy_teachers, value_teachers);
}

std::array<float, 2> validation(const std::vector<std::pair<string, TeacherType>>& validation_data) {
    static constexpr int32_t batch_size = 4096;
    int32_t num = 0;
    float policy_loss = 0.0, value_loss = 0.0;
    Position pos;
    while (num < validation_data.size()) {
        std::vector<float> inputs;
        std::vector<PolicyTeacherType> policy_teachers;
        std::vector<ValueTeacherType> value_teachers;

        //ミニバッチ分貯める
        //一番最後ではミニバッチ数ピッタリにならないかもしれないのでカウントする
        while (num < validation_data.size() && policy_teachers.size() < batch_size) {
            const auto& datum = validation_data[num++];
            pos.loadSFEN(datum.first);
            const auto feature = pos.makeFeature();
            inputs.insert(inputs.end(), feature.begin(), feature.end());
            policy_teachers.push_back(datum.second.policy);
            value_teachers.push_back(datum.second.value);
        }

        //計算
        auto val_loss = nn->loss(inputs, policy_teachers, value_teachers);

        //平均化されて返ってくるのでバッチサイズをかけて総和に戻す
        //一番最後はbatch_sizeピッタリになるとは限らないのでちゃんとサイズを見てかける値を決める
        auto curr_size = policy_teachers.size();
#ifdef USE_LIBTORCH
        policy_loss += val_loss.first.item<float>() * curr_size;
        value_loss  += val_loss.second.item<float>() * curr_size;
#else
        policy_loss += val_loss.first.to_float() * curr_size;
        value_loss  += val_loss.second.to_float() * curr_size;
#endif
    }
    policy_loss /= validation_data.size();
    value_loss /= validation_data.size();

    return { policy_loss, value_loss };
}

std::vector<std::pair<string, TeacherType>> loadData(const std::string& file_path) {
    //棋譜を読み込めるだけ読み込む
    auto games = loadGames(file_path, 100000);

    //データを局面単位にバラす
    std::vector<std::pair<std::string, TeacherType>> data_buffer;
    for (const auto& game : games) {
        Position pos;
        for (const auto& move : game.moves) {
            TeacherType teacher;
            teacher.policy = (uint32_t) move.toLabel();
#ifdef USE_CATEGORICAL
            assert(false);
            teacher.value = valueToIndex((pos.color() == BLACK ? game.result : MAX_SCORE + MIN_SCORE - game.result));
#else
            teacher.value = (float) (pos.color() == BLACK ? game.result : MAX_SCORE + MIN_SCORE - game.result);
#endif
            data_buffer.emplace_back(pos.toSFEN(), teacher);
            pos.doMove(move);
        }
    }

    return data_buffer;
}