﻿#ifndef MIACIS_NEURAL_NETWORK_HPP
#define MIACIS_NEURAL_NETWORK_HPP

#include "neural_network_modules.hpp"
#include "types.hpp"

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

//損失の種類
enum LossType { POLICY_LOSS_INDEX, VALUE_LOSS_INDEX, LOSS_TYPE_NUM };

//各損失の名前を示す文字列
const std::array<std::string, LOSS_TYPE_NUM> LOSS_TYPE_NAME{ "policy", "value" };

//#define REPRESENTATION_DROPOUT

//使用する全体のニューラルネットワーク
class NeuralNetworkImpl : public torch::nn::Module {
public:
    NeuralNetworkImpl();

    //入力として局面の特徴量を並べたvectorを受け取ってPolicyとValueに対応するTensorを返す関数
    std::pair<torch::Tensor, torch::Tensor> forward(const std::vector<float>& inputs);

    //複数局面の特徴量を1次元vectorにしたものを受け取ってそれぞれに対する評価を返す関数
    std::pair<std::vector<PolicyType>, std::vector<ValueType>> policyAndValueBatch(const std::vector<float>& inputs);

    //学習データについて損失を返す関数
    std::array<torch::Tensor, LOSS_TYPE_NUM> loss(const std::vector<LearningData>& data);

    //データに対して検証損失を返す関数
    std::array<torch::Tensor, LOSS_TYPE_NUM> validLoss(const std::vector<LearningData>& data);

    //GPUにネットワークを送る関数
    void setGPU(int16_t gpu_id, bool fp16 = false);

    //評価パラメータを読み書きするファイルのprefix
    static const std::string MODEL_PREFIX;

    //デフォルトで読み書きするファイル名
    static const std::string DEFAULT_MODEL_NAME;

private:
    torch::Tensor encode(const std::vector<float>& inputs);
    std::pair<torch::Tensor, torch::Tensor> decode(const torch::Tensor& representation);

    torch::Device device_;
    bool fp16_;

    Conv2DwithBatchNorm state_first_conv_and_norm_{ nullptr };
    std::vector<ResidualBlock> state_blocks_;

#ifdef REPRESENTATION_DROPOUT
    torch::nn::Dropout2d representation_dropout_{ nullptr };
#endif

    torch::nn::Conv2d policy_conv_{ nullptr };
    Conv2DwithBatchNorm value_conv_and_norm_{ nullptr };
    torch::nn::Linear value_linear0_{ nullptr };
    torch::nn::Linear value_linear1_{ nullptr };
};
TORCH_MODULE(NeuralNetwork);

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

#endif //MIACIS_NEURAL_NETWORK_HPP