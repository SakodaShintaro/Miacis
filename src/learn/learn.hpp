#ifndef MIACIS_LEARN_HPP
#define MIACIS_LEARN_HPP

#include "../model/learning_model.hpp"
#include "../model/model_common.hpp"
#include "../model/tensorrt_model.hpp"
#include "../model/torch_tensorrt_model.hpp"
#include "../timer.hpp"
#include <torch/torch.h>

//教師データを読み込む関数
std::vector<LearningData> loadData(const std::string& file_path, bool data_augmentation, float rate_threshold);
std::vector<LearningData> loadHCPE(const std::string& file_path, bool data_augmentation);

//validationを行う関数
std::array<float, LOSS_TYPE_NUM> validation(TensorRTModel& model, const std::vector<LearningData>& valid_data,
                                            uint64_t batch_size);
std::array<float, LOSS_TYPE_NUM> validation(LearningModel& model, const std::vector<LearningData>& valid_data,
                                            uint64_t batch_size);
std::array<float, LOSS_TYPE_NUM> validation(TorchTensorRTModel& model, const std::vector<LearningData>& valid_data,
                                            uint64_t batch_size);

//validationを行い、各局面の損失をtsvで出力する関数
std::array<float, LOSS_TYPE_NUM> validationWithSave(TensorRTModel& model, const std::vector<LearningData>& valid_data,
                                                    uint64_t batch_size);

//学習データをtensorへ変換する関数
torch::Tensor getInputTensor(const std::vector<LearningData>& data, torch::Device device);
torch::Tensor getPolicyTargetTensor(const std::vector<LearningData>& data, torch::Device device);
torch::Tensor getValueTargetTensor(const std::vector<LearningData>& data, torch::Device device);
torch::Tensor getCategoricalValueTargetTensor(const std::vector<LearningData>& data, torch::Device device);

// 強化学習を行う関数
void reinforcementLearn();

#endif //MIACIS_LEARN_HPP