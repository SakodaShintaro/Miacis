#include "dataset.hpp"
#include "include_switch.hpp"
#include "learn.hpp"
#include "neural_network.hpp"

MyDataset::MyDataset(const std::string& root) {
    std::vector<LearningData> data = loadData(root, false, 3200);
    Position pos;
    std::vector<ValueTeacherType> value_teachers;

    for (const LearningData& datum : data) {
        pos.fromStr(datum.position_str);

        //入力
        const std::vector<float> feature = pos.makeFeature();
        std::vector<float> inputs;
        inputs.insert(inputs.end(), feature.begin(), feature.end());
        data_.push_back(torch::tensor(inputs).view({ 1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH }));

        //policyの教師信号
        std::vector<float> policy_teachers(POLICY_DIM, 0.0);
        for (const std::pair<int32_t, float>& e : datum.policy) {
            policy_teachers[e.first] = e.second;
        }
        targets_.push_back(torch::tensor(policy_teachers));

        //valueの教師信号
        value_teachers.push_back(datum.value);
    }
}

torch::data::Example<> MyDataset::get(size_t index) {
    return { data_[index].clone().to(torch::kCUDA), targets_[index].clone().to(torch::kCUDA) };
}

c10::optional<size_t> MyDataset::size() const { return data_.size(); }