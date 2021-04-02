#include "dataset.hpp"
#include "include_switch.hpp"
#include "learn.hpp"
#include "neural_network.hpp"

CalibrationDataset::CalibrationDataset(const std::string& root, int64_t data_num) {
    std::vector<LearningData> data = loadData(root, false, 3200);
    Position pos;

    for (const LearningData& datum : data) {
        pos.fromStr(datum.position_str);

        //入力
        std::vector<float> inputs = pos.makeFeature();
        data_.push_back(torch::tensor(inputs).view({ 1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH }));

        //targetの方は使わないのでダミーの適当な値を入れる
        targets_.push_back(torch::tensor({ 0 }));

        //全データだと多いので、先頭からいくつかのみを用いる
        if (data_.size() >= data_num) {
            break;
        }
    }
}

torch::data::Example<> CalibrationDataset::get(size_t index) {
    return { data_[index].clone().to(torch::kCUDA), targets_[index].clone().to(torch::kCUDA) };
}

c10::optional<size_t> CalibrationDataset::size() const { return data_.size(); }