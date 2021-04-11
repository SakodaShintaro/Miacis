#ifndef MIACIS_NEURAL_NETWORK_HPP
#define MIACIS_NEURAL_NETWORK_HPP

#include "types.hpp"
#include <torch/torch.h>

//型のエイリアス
using PolicyType = std::vector<float>;
using PolicyTeacherType = std::vector<std::pair<int32_t, float>>;
#ifdef USE_CATEGORICAL
constexpr int32_t BIN_SIZE = 51;
constexpr float VALUE_WIDTH = (MAX_SCORE - MIN_SCORE) / BIN_SIZE;
using ValueType = std::array<float, BIN_SIZE>;
using ValueTeacherType = int64_t;
#else
constexpr int32_t BIN_SIZE = 1;
using ValueType = float;
using ValueTeacherType = float;
#endif

//学習データの型
struct LearningData {
    std::string position_str;
    PolicyTeacherType policy;
    ValueTeacherType value;
};

extern const std::string MODEL_PREFIX;
extern const std::string DEFAULT_MODEL_NAME;

//損失の種類
enum LossType { POLICY_LOSS_INDEX, VALUE_LOSS_INDEX, LOSS_TYPE_NUM };

//各損失の名前を示す文字列
const std::array<std::string, LOSS_TYPE_NUM> LOSS_TYPE_NAME{ "policy", "value" };

//推論時の補助関数
//CPU上にあるTensorのペア(それぞれPolicy, Value)をstd::vector<へ変換する
std::pair<std::vector<PolicyType>, std::vector<ValueType>> tensorToVector(const std::tuple<torch::Tensor, torch::Tensor>& output,
                                                                          bool use_fp16);

//Categorical分布に対する操作
#ifdef USE_CATEGORICAL
inline int32_t valueToIndex(float value) { return std::min((int32_t)((value - MIN_SCORE) / VALUE_WIDTH), BIN_SIZE - 1); }

inline ValueType onehotDist(float value) {
    //valueForBlackのところだけ1.0, 他は0.0とした分布を返す
    ValueType result{};
    result[valueToIndex(value)] = 1.0;
    return result;
}

inline float expOfValueDist(ValueType dist) {
    float exp = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        //i番目の要素が示す値はMIN_SCORE + (i + 0.5) * VALUE_WIDTH
        exp += (MIN_SCORE + (i + 0.5) * VALUE_WIDTH) * dist[i];
    }
    return exp;
}
#endif

// dlshogiの設定
constexpr int64_t DLSHOGI_FEATURES1_NUM = 62;
constexpr int64_t DLSHOGI_FEATURES2_NUM = 57;

#endif //MIACIS_NEURAL_NETWORK_HPP