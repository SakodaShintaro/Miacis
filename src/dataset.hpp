#ifndef MIACIS_DATASET_HPP
#define MIACIS_DATASET_HPP

#include <torch/torch.h>

// The MyDataset Dataset
class MyDataset : public torch::data::datasets::Dataset<MyDataset> {
public:
    explicit MyDataset(const std::string& root);

    // Returns the pair at index in the dataset
    torch::data::Example<> get(size_t index) override;

    // The size of the dataset
    c10::optional<size_t> size() const override;

private:
    std::vector<torch::Tensor> data_, targets_;
};

#endif //MIACIS_DATASET_HPP