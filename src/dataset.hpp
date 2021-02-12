#ifndef MIACIS_DATASET_HPP
#define MIACIS_DATASET_HPP

#include <torch/torch.h>

class CalibrationDataset : public torch::data::datasets::Dataset<CalibrationDataset> {
public:
    explicit CalibrationDataset(const std::string& root);

    torch::data::Example<> get(size_t index) override;

    c10::optional<size_t> size() const override;

private:
    std::vector<torch::Tensor> data_, targets_;
};

#endif //MIACIS_DATASET_HPP