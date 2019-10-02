#ifndef MIACIS_NEURAL_NETWORK_HPP
#define MIACIS_NEURAL_NETWORK_HPP

#include"position.hpp"
#include<torch/torch.h>

//ネットワーク構造によらない定数
constexpr int32_t POLICY_CHANNEL_NUM = 27;
constexpr int32_t POLICY_DIM = SQUARE_NUM * POLICY_CHANNEL_NUM;

//型のエイリアス
using FloatType = float;
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
    Move move;
    ValueTeacherType value;
};

//損失の種類
enum LossType {
    POLICY_LOSS_INDEX, VALUE_LOSS_INDEX, TRANS_LOSS_INDEX, LOSS_TYPE_NUM
};

//畳み込みとBatchNormalizationをまとめたユニット
class Conv2DwithBatchNormImpl : public torch::nn::Module {
public:
    Conv2DwithBatchNormImpl(int64_t input_ch, int64_t output_ch, int64_t kernel_size);
    torch::Tensor forward(const torch::Tensor& x);
private:
    torch::nn::Conv2d    conv_{ nullptr };
    torch::nn::BatchNorm norm_{ nullptr };
};
TORCH_MODULE(Conv2DwithBatchNorm);

//残差ブロック:SENetの構造を利用
class ResidualBlockImpl : public torch::nn::Module {
public:
    ResidualBlockImpl(int64_t channel_num, int64_t kernel_size, int64_t reduction);
    torch::Tensor forward(const torch::Tensor& x);
private:
    Conv2DwithBatchNorm conv_and_norm0_{ nullptr };
    Conv2DwithBatchNorm conv_and_norm1_{ nullptr };
    torch::nn::Linear   linear0_{ nullptr };
    torch::nn::Linear   linear1_{ nullptr };
};
TORCH_MODULE(ResidualBlock);

//使用する全体のニューラルネットワーク
class NeuralNetworkImpl : public torch::nn::Module {
public:
    NeuralNetworkImpl();

    //複数局面の特徴量を1次元vectorにしたものを受け取ってそれぞれに対する評価を返す関数
    std::pair<std::vector<PolicyType>, std::vector<ValueType>> policyAndValueBatch(const std::vector<float>& inputs);

    //ある局面の表現を1次元vectorにしたものを受け取ってそれに対する評価を返す関数
    std::pair<std::vector<PolicyType>, std::vector<ValueType>> decodePolicyAndValueBatch(const std::vector<float>& state_rep);

    //データから損失を計算する関数
    std::array<torch::Tensor, LOSS_TYPE_NUM> loss(const std::vector<LearningData>& data);

    //このネットワークが計算されるGPU or CPUを設定する関数
    void setGPU(int16_t gpu_id, bool fp16 = false);

    //状態を表現へと変換する関数
    torch::Tensor encodeStates(const std::vector<float>& inputs);

    //行動を表現へと変換する関数
    torch::Tensor encodeActions(const std::vector<Move>& moves);

    //状態表現から方策を得る関数
    torch::Tensor decodePolicy(const torch::Tensor& representation);

    //状態表現から状態価値を得る関数
    torch::Tensor decodeValue(const torch::Tensor& representation);

    //状態表現と行動表現から次状態の表現を予測する関数
    torch::Tensor predictTransition(const torch::Tensor& state_representations, const torch::Tensor& move_representations);

    //状態表現(std::vector)と行動(Move一つ)から次状態の表現を予測する関数
    std::vector<FloatType> predictTransition(const std::vector<FloatType>& state_rep, Move move);

    //状態表現の予測と実際の表現から損失を計算する関数
    static torch::Tensor transitionLoss(const torch::Tensor& predict, const torch::Tensor& ground_truth);

    //評価パラメータを読み書きするファイルのprefix
    static const std::string MODEL_PREFIX;

    //デフォルトで読み書きするファイル名
    static const std::string DEFAULT_MODEL_NAME;

private:
    //このネットワークが計算されるGPU or CPU
    torch::Device device_;
    bool fp16_;

    //encodeStateで使用
    Conv2DwithBatchNorm state_encoder_first_conv_and_norm_{nullptr};
    std::vector<ResidualBlock> state_encoder_blocks_;

    //encodeActionで使用
    Conv2DwithBatchNorm action_encoder_first_conv_and_norm_{nullptr};
    std::vector<ResidualBlock> action_encoder_blocks_;

    //predictTransitionで使用
    Conv2DwithBatchNorm predict_transition_first_conv_and_norm_{nullptr};
    std::vector<ResidualBlock> predict_transition_blocks_;

    //decodePolicyで使用
    torch::nn::Conv2d policy_conv_{nullptr};

    //decodeValueで使用
    Conv2DwithBatchNorm value_conv_and_norm_{nullptr};
    torch::nn::Linear value_linear0_{nullptr};
    torch::nn::Linear value_linear1_{nullptr};
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