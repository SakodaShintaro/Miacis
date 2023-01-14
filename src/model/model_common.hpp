#ifndef MIACIS_MODEL_COMMON_HPP
#define MIACIS_MODEL_COMMON_HPP

#include "../types.hpp"
#include <torch/torch.h>
#include <vector>

//型のエイリアス
using PolicyType = std::vector<float>;
using PolicyTeacherType = std::vector<std::pair<int32_t, float>>;
#ifdef USE_CATEGORICAL
constexpr int32_t BIN_SIZE = 51;
constexpr float VALUE_WIDTH = (MAX_SCORE - MIN_SCORE) / BIN_SIZE;
using ValueType = std::array<float, BIN_SIZE>;
#else
constexpr int32_t BIN_SIZE = 1;
using ValueType = float;
#endif
using ValueTeacherType = float;

//学習データの型
struct LearningData {
    std::string position_str;
    PolicyTeacherType policy;
    ValueTeacherType value;
};

extern const std::string DEFAULT_MODEL_PREFIX;
extern const std::string DEFAULT_MODEL_NAME;
extern const std::string DEFAULT_ONNX_NAME;
extern const std::string DEFAULT_ENGINE_NAME;

//損失の種類
enum LossType { POLICY_LOSS_INDEX, VALUE_LOSS_INDEX, LOSS_TYPE_NUM };

//各損失の名前を示す文字列
const std::array<std::string, LOSS_TYPE_NUM> LOSS_TYPE_NAME{ "policy", "value" };

//入力のvectorをTensorに変換する関数
torch::Tensor inputVectorToTensor(const std::vector<float>& input);

//Categorical分布に対する操作
#ifdef USE_CATEGORICAL
inline int32_t valueToIndex(float value) { return std::min((int32_t)((value - MIN_SCORE) / VALUE_WIDTH), BIN_SIZE - 1); }
inline float indexToValue(int32_t index) { return (MIN_SCORE + (index + 0.5) * VALUE_WIDTH); }

inline ValueType onehotDist(float value) {
    //valueForBlackのところだけ1.0, 他は0.0とした分布を返す
    ValueType result{};
    result[valueToIndex(value)] = 1.0;
    return result;
}

inline float expOfValueDist(ValueType dist) {
    float exp = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        exp += indexToValue(i) * dist[i];
    }
    return exp;
}
#else

inline float expOfValueDist(ValueType dist) { return dist; }
#endif

#endif //MIACIS_MODEL_COMMON_HPP