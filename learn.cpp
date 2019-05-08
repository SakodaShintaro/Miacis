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

std::array<float, 2> validation(const std::vector<std::pair<std::string, TeacherType>>& validation_data) {
    static constexpr int32_t batch_size = 4096;
    int32_t index = 0;
    float policy_loss = 0.0, value_loss = 0.0;
    torch::NoGradGuard no_grad_guard;
    Position pos;
    while (index < validation_data.size()) {
        std::vector<float> inputs;
        std::vector<PolicyTeacherType> policy_teachers;
        std::vector<ValueTeacherType> value_teachers;
        inputs.reserve(batch_size * INPUT_CHANNEL_NUM);
        policy_teachers.reserve(batch_size);
        value_teachers.reserve(batch_size);

        //バッチサイズ分データを確保
        while (index < validation_data.size() && policy_teachers.size() < batch_size) {
            const auto& datum = validation_data[index++];
            pos.loadSFEN(datum.first);
            const auto feature = pos.makeFeature();
            inputs.insert(inputs.end(), feature.begin(), feature.end());
            policy_teachers.push_back(datum.second.policy);
            value_teachers.push_back(datum.second.value);
        }

        //計算
        auto loss = nn->loss(inputs, policy_teachers, value_teachers);

        policy_loss += loss.first.sum().item<float>();

#ifdef USE_CATEGORICAL
        //categoricalモデルのときは冗長だがもう一度順伝播を行って損失を手動で計算
        auto y = nn->policyAndValueBatch(inputs);
        const auto& values = y.second;
        for (int32_t i = 0; i < values.size(); i++) {
            auto e = expOfValueDist(values[i]);
            auto vt = (value_teachers[i] == BIN_SIZE - 1 ? MAX_SCORE : MIN_SCORE);
            value_loss += (e - vt) * (e - vt);
        }
#else
        //scalarモデルのときはそのまま損失を加える
        value_loss += loss.second.sum().item<float>();
#endif
    }

    //平均を求める
    policy_loss /= validation_data.size();
    value_loss /= validation_data.size();

    return { policy_loss, value_loss };
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