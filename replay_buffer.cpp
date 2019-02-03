#include"replay_buffer.hpp"
#include"operate_params.hpp"
#include "replay_buffer.hpp"

#include<thread>

std::tuple<std::vector<float>, std::vector<uint32_t>, std::vector<ValueTeacher>> ReplayBuffer::makeBatch(int32_t batch_size) {
    //ロックの確保する必要あるかな？
    std::unique_lock<std::mutex> lock(mutex_);

    //とりあえずランダムに取得
    std::mt19937 engine(0);
    std::uniform_int_distribution<unsigned long> dist(0, data_.size() - 1);

    Position pos;

    std::vector<float> inputs;
    std::vector<uint32_t> policy_labels;
    std::vector<ValueTeacher> value_teachers;
    for (int32_t i = 0; i < batch_size; i++) {
        std::string sfen;
        uint32_t policy_label;
        float value;
        const auto& datum = data_[dist(engine)];
        std::tie(sfen, policy_label, value) = datum;

        pos.loadSFEN(sfen);
        for (const auto& e : pos.makeFeature()) {
            inputs.push_back(e);
        }
        policy_labels.push_back(policy_label);

#ifdef USE_CATEGORICAL
        value_teachers.push_back(valueToIndex(value));
#else
        value_teachers.push_back(value);
#endif
    }

    return std::make_tuple(inputs, policy_labels, value_teachers);
}

void ReplayBuffer::push(const Position &pos, Move move, float value) {
    std::unique_lock<std::mutex> lock(mutex_);
    data_.emplace_back(pos.toSFEN(), move.toLabel(), value);
}

void ReplayBuffer::setSize(int64_t max_size) {
    max_size_ = max_size;
    data_.reserve(max_size);
}

void ReplayBuffer::push(const Position &pos, TeacherType teacher) {
    std::unique_lock<std::mutex> lock(mutex_);
    data_.emplace_back(pos.toSFEN(), teacher.policy, teacher.value);
}

void ReplayBuffer::push(const std::string &sfen, TeacherType teacher) {
    std::unique_lock<std::mutex> lock(mutex_);
    data_.emplace_back(sfen, teacher.policy, teacher.value);
}

void ReplayBuffer::show() {
    for (const auto& data : data_) {
        std::string sfen;
        uint32_t policy_label;
        ValueTeacher value_teacher;
        std::tie(sfen, policy_label, value_teacher) = data;
        Position pos;
        pos.loadSFEN(sfen);
        pos.print();
    }
}
