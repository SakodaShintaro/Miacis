#ifndef MIACIS_LEARN_HPP
#define MIACIS_LEARN_HPP

#include "../model/infer_model.hpp"
#include "../model/model_common.hpp"
#include "../timer.hpp"

//教師データを読み込む関数
std::vector<LearningData> loadData(const std::string& file_path, bool data_augmentation, float rate_threshold);
std::vector<LearningData> loadHCPE(const std::string& file_path, bool data_augmentation);

//validationを行う関数
std::array<float, LOSS_TYPE_NUM> validation(InferModel& model, const std::vector<LearningData>& valid_data, uint64_t batch_size);

//validationを行い、各局面の損失をtsvで出力する関数
std::array<float, LOSS_TYPE_NUM> validationWithSave(InferModel& model, const std::vector<LearningData>& valid_data, uint64_t batch_size);

#endif //MIACIS_LEARN_HPP