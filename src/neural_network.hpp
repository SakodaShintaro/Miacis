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
using PolicyTeacherType = std::vector<std::pair<int32_t, float>>;

//学習データの型
struct LearningData {
    std::string SFEN;
    Move move;
    ValueTeacherType value;
};

enum LossOrder {
    POLICY, VALUE, TRANS, LOSS_NUM
};

class NeuralNetworkImpl : public torch::nn::Module {
public:
    NeuralNetworkImpl();
    //状態を表現へと変換する関数
    torch::Tensor encodeState(const std::vector<float>& inputs);

    //行動を表現へと変換する関数
    torch::Tensor encodeAction(const std::vector<Move>& moves);

    //状態表現から方策を得る関数
    torch::Tensor decodePolicy(torch::Tensor representation);

    //状態表現から状態価値を得る関数
    torch::Tensor decodeValue(torch::Tensor representation);

    //複数局面の特徴量を1次元vectorにしたものを受け取ってそれぞれに対する評価を返す関数
    std::pair<std::vector<PolicyType>, std::vector<ValueType>> policyAndValueBatch(const std::vector<float>& inputs);

    //状態表現と行動から次状態の表現を予測する関数
    torch::Tensor predictNextStateRep(torch::Tensor state_representations, torch::Tensor move_representations);

    std::array<torch::Tensor, LOSS_NUM> loss(const std::vector<LearningData>& data);

    void setGPU(int16_t gpu_id);

private:
    static constexpr int64_t REPRESENTATION_DIM = CHANNEL_NUM;

    torch::Device device_;

    //encodeStateで使用
    torch::nn::Conv2d    first_conv{nullptr};
    torch::nn::BatchNorm first_norm{nullptr};
    std::vector<std::vector<torch::nn::Conv2d>>    conv;
    std::vector<std::vector<torch::nn::BatchNorm>> norm;
    std::vector<std::vector<torch::nn::Linear>>    fc;

    //encodeActionで使用
    torch::nn::Linear action_encode_fc{nullptr};

    //decodePolicyで使用
    torch::nn::Linear policy_fc{nullptr};

    //decodeValueで使用
    torch::nn::Linear value_fc1{nullptr};
    torch::nn::Linear value_fc2{nullptr};
    torch::nn::Linear value_fc3{nullptr};

    //predictNextStateRepで使用
    torch::nn::Linear next_state_rep_predictor{nullptr};
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