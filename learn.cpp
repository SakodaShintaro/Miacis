#include"learn.hpp"
#include"game.hpp"
#include"operate_params.hpp"
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

#ifdef USE_CATEGORICAL
std::array<float, 3> validation(const std::vector<std::pair<std::string, TeacherType>>& validation_data) {
#else
std::array<float, 2> validation(const std::vector<std::pair<std::string, TeacherType>>& validation_data) {
#endif
    static constexpr int32_t batch_size = 4096;
    int32_t num = 0;
    float policy_loss = 0.0, value_loss = 0.0;
#ifdef USE_CATEGORICAL
    float value_loss2 = 0.0;
#endif
    torch::NoGradGuard no_grad_guard;
    Position pos;
    while (num < validation_data.size()) {
        std::vector<float> inputs;
        std::vector<PolicyTeacherType> policy_teachers;
        std::vector<ValueTeacherType> value_teachers;
        inputs.reserve(batch_size * INPUT_CHANNEL_NUM);
        policy_teachers.reserve(batch_size);
        value_teachers.reserve(batch_size);

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
        auto loss = nn->loss(inputs, policy_teachers, value_teachers);

#ifdef USE_CATEGORICAL
        auto y = nn->policyAndValueBatch(inputs);
        const auto& values = y.second;
        for (int32_t i = 0; i < values.size(); i++) {
            auto e = expOfValueDist(values[i]);
            auto vt = (value_teachers[i] == BIN_SIZE - 1 ? MAX_SCORE : MIN_SCORE);
            value_loss2 += (e - vt) * (e - vt);
        }
#endif

        //平均化されて返ってくるのでバッチサイズをかけて総和に戻す
        //一番最後はbatch_sizeピッタリになるとは限らないのでちゃんとサイズを見てかける値を決める
        auto curr_size = policy_teachers.size();
        policy_loss += loss.first.item<float>() * curr_size;
        value_loss  += loss.second.item<float>() * curr_size;
    }
    policy_loss /= validation_data.size();
    value_loss /= validation_data.size();

#ifdef USE_CATEGORICAL
    value_loss2 /= validation_data.size();
    return { policy_loss, value_loss, value_loss2 };
#else
    return { policy_loss, value_loss };
#endif
}

std::vector<std::pair<std::string, TeacherType>> loadData(const std::string& file_path) {
    //棋譜を読み込めるだけ読み込む
    auto games = loadGames(file_path, 100000);

    //データを局面単位にバラす
    std::vector<std::pair<std::string, TeacherType>> data_buffer;
    for (const auto& game : games) {
        Position pos;
        for (const auto& e : game.elements) {
            const auto& move = e.move;
            TeacherType teacher;
            teacher.policy = (uint32_t) move.toLabel();
#ifdef USE_CATEGORICAL
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