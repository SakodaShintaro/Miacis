#include"learn.hpp"
#include "learn.hpp"

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
