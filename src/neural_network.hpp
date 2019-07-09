#ifndef MIACIS_NEURAL_NETWORK_HPP
#define MIACIS_NEURAL_NETWORK_HPP

#include"position.hpp"
#include<torch/torch.h>

//ネットワークの設定
constexpr int32_t POLICY_CHANNEL_NUM = 27;
constexpr int32_t BLOCK_NUM = 10;
constexpr int32_t KERNEL_SIZE = 3;
constexpr int32_t CHANNEL_NUM = 64;
constexpr int32_t REDUCTION = 8;
constexpr int32_t VALUE_HIDDEN_NUM = 256;

//評価パラメータを読み書きするファイルのprefix
#ifdef USE_CATEGORICAL
const std::string MODEL_PREFIX = "cat_bl" + std::to_string(BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
#else
const std::string MODEL_PREFIX = "sca_bl" + std::to_string(BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
#endif
//デフォルトで読み書きするファイル名
const std::string MODEL_PATH = MODEL_PREFIX + ".model";

//型のエイリアス
using CalcType = float;
using PolicyType = std::vector<float>;
using PolicyTeacherType = std::vector<std::pair<int32_t, float>>;
#ifdef USE_CATEGORICAL
constexpr int32_t BIN_SIZE = 51;
constexpr double VALUE_WIDTH = (MAX_SCORE - MIN_SCORE) / BIN_SIZE;
using ValueType = std::array<float, BIN_SIZE>;
using ValueTeacherType = int64_t;
#else
constexpr int32_t BIN_SIZE = 1;
using ValueType = float;
using ValueTeacherType = float;
#endif

//学習データの型
struct LearningData {
    std::string SFEN;
    PolicyTeacherType policy;
    ValueTeacherType value;
};

//損失の種類
enum LossType {
    POLICY_LOSS_INDEX, VALUE_LOSS_INDEX, LOSS_TYPE_NUM
};

class NeuralNetworkImpl : public torch::nn::Module {
public:
    NeuralNetworkImpl();

    //入力としてvectorを受け取ってTensorを返す関数
    std::pair<torch::Tensor, torch::Tensor> forward(const std::vector<float>& inputs);

    //複数局面の特徴量を1次元vectorにしたものを受け取ってそれぞれに対する評価を返す関数
    std::pair<std::vector<PolicyType>, std::vector<ValueType>> policyAndValueBatch(const std::vector<float>& inputs);

    //バッチの入力特徴量,教師情報を引数として損失を返す関数.これをモデルが一括で行うのが良い実装？
    std::array<torch::Tensor, LOSS_TYPE_NUM> loss(const std::vector<LearningData>& data);

    void setGPU(int16_t gpu_id);

private:
    torch::Device device_;
    torch::nn::Conv2d first_conv{nullptr};
    torch::nn::BatchNorm first_bn{nullptr};
    std::vector<std::vector<torch::nn::Conv2d>> conv;
    std::vector<std::vector<torch::nn::BatchNorm>> bn;
    std::vector<std::vector<torch::nn::Linear>> fc;
    torch::nn::Conv2d policy_conv{nullptr};
    torch::nn::Conv2d value_conv{nullptr};
    torch::nn::BatchNorm value_bn{nullptr};
    torch::nn::Linear value_fc1{nullptr};
    torch::nn::Linear value_fc2{nullptr};
};
TORCH_MODULE(NeuralNetwork);

extern NeuralNetwork nn;

//Categorical分布に対する操作
#ifdef USE_CATEGORICAL
inline int32_t valueToIndex(double value) {
    return std::min((int32_t)((value - MIN_SCORE) / VALUE_WIDTH), BIN_SIZE - 1);
}

inline ValueType onehotDist(double value) {
    //valueForBlackのところだけ1.0, 他は0.0とした分布を返す
    ValueType result{};
    result[valueToIndex(value)] = 1.0;
    return result;
}

inline double expOfValueDist(ValueType dist) {
    double exp = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        //i番目の要素が示す値はMIN_SCORE + (i + 0.5) * VALUE_WIDTH
        exp += (MIN_SCORE + (i + 0.5) * VALUE_WIDTH) * dist[i];
    }
    return exp;
}
#endif

#endif //MIACIS_NEURAL_NETWORK_HPP