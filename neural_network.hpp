#ifndef MIACIS_NEURAL_NETWORK_HPP
#define MIACIS_NEURAL_NETWORK_HPP

#include"position.hpp"
#include<vector>

//ネットワークの設定
constexpr int32_t POLICY_CHANNEL_NUM = 27;
constexpr int32_t BLOCK_NUM = 4;
constexpr int32_t KERNEL_SIZE = 3;
constexpr int32_t CHANNEL_NUM = 32;
constexpr int32_t VALUE_HIDDEN_NUM = 256;

#ifdef USE_LIBTORCH
//LibTorchを使う
#include<torch/torch.h>

//評価パラメータを読み書きするファイルのprefix
#ifdef USE_CATEGORICAL
const std::string MODEL_PREFIX = "torch_cat_bl" + std::to_string(BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
#else
const std::string MODEL_PREFIX = "torch_sca_bl" + std::to_string(BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
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
using PolicyTeacherType = int64_t;

extern torch::Device device;

class NeuralNetworkImpl : public torch::nn::Module {
public:
    NeuralNetworkImpl();

    //Tensorを受け取ってpolicyとvalueに相当するTensorを返す関数
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

    //上の関数をラップして入力部をfloatのvectorにした関数.不要かもしれない
    std::pair<torch::Tensor, torch::Tensor> forward(const std::vector<float>& inputs);

    //1局面を受け取ってそれに対する評価を返す関数.不要な気がする
    std::pair<PolicyType, ValueType> policyAndValue(const Position& pos);

    //複数局面の特徴量を1次元vectorにしたものを受け取ってそれぞれに対する評価を返す関数
    std::pair<std::vector<PolicyType>, std::vector<ValueType>> policyAndValueBatch(const std::vector<float>& inputs);

    //バッチの入力特徴量,教師情報を引数として損失を返す関数.これをモデルが一括で行うのが良い実装？
    std::pair<torch::Tensor, torch::Tensor> loss(const std::vector<float>& input,
                                                 const std::vector<PolicyTeacherType>& policy_teachers,
                                                 const std::vector<ValueTeacherType>& value_teachers);

private:
    torch::nn::Conv2d first_conv{nullptr};
    torch::nn::BatchNorm first_bn{nullptr};
    std::vector<std::vector<torch::nn::Conv2d>> conv;
    std::vector<std::vector<torch::nn::BatchNorm>> bn;
    torch::nn::Conv2d policy_conv{nullptr};
    torch::nn::Conv2d value_conv{nullptr};
    torch::nn::BatchNorm value_bn{nullptr};
    torch::nn::Linear value_fc1{nullptr};
    torch::nn::Linear value_fc2{nullptr};
};
TORCH_MODULE(NeuralNetwork);

extern NeuralNetwork nn;

#else
//primitivを使う
#include<primitiv/primitiv.h>

//評価パラメータを読み書きするファイルのprefix
#ifdef USE_CATEGORICAL
const std::string MODEL_PREFIX = "primitiv_cat_bl" + std::to_string(BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
#else
const std::string MODEL_PREFIX = "primitiv_sca_bl" + std::to_string(BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
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
using ValueTeacherType = uint32_t;
#else
constexpr int32_t BIN_SIZE = 1;
using ValueType = float;
using ValueTeacherType = float;
#endif
using PolicyTeacherType = uint32_t;

using namespace primitiv;
namespace F = primitiv::functions;
namespace I = primitiv::initializers;
namespace O = primitiv::optimizers;

template <typename Var>
class BatchNormLayer : public primitiv::Model {
public:
    BatchNormLayer();

    //Shapeに合わせたパラメータを確保する関数
    void init(Shape s);

    //下2つはそれぞれテンプレートの特殊化で書くべきか
    //バッチ平均を計算してそれを用いて移動平均を計算する関数
    Node   operator()(Node x);

    //計算された移動平均を用いて計算する関数
    Tensor operator()(Tensor x);
private:
    Parameter beta_, gamma_, mean_, var_;
};

template <typename Var>
class NeuralNetwork : public primitiv::Model {
public:
    NeuralNetwork();

    //モデルへのパラメータ登録,パラメータの初期化などを行う関数
    void init();

    //複数バッチ分の入力特徴量を1次元vectorとしたものを引数としてバッチ数のデータを持つVarを返す関数.privateでいいか？
    std::pair<Var, Var> feedForward(const std::vector<float>& input);

    //1局面を受け取って評価結果を返す関数.不要そう
    std::pair<PolicyType, ValueType> policyAndValue(const Position& pos);

    //複数バッチ分の入力特徴量を1次元vectorとしたものを引数としてそれぞれの評価結果を返す関数.内部でfeedForwardを呼び出す
    std::pair<std::vector<PolicyType>, std::vector<ValueType>> policyAndValueBatch(const std::vector<float>& inputs);

    //複数バッチ分の入力特徴量,教師情報から損失を返す関数.F::batch::meanをどこのタイミングでかけるべきか
    std::pair<Var, Var> loss(const std::vector<float>& input, const std::vector<uint32_t>& policy_labels,
                             const std::vector<ValueTeacherType>& value_teachers);

private:
    Parameter first_filter;
    Parameter filter[BLOCK_NUM][2];
    Parameter policy_filter;
    Parameter value_filter;
    Parameter value_pw_fc1;
    Parameter value_pb_fc1;
    Parameter value_pw_fc2;
    Parameter value_pb_fc2;

    BatchNormLayer<Var> first_bn;
    BatchNormLayer<Var> bn[BLOCK_NUM][2];
    BatchNormLayer<Var> value_bn;
};

extern std::unique_ptr<NeuralNetwork<Tensor>> nn;

#endif

//教師の型
struct TeacherType {
    PolicyTeacherType policy;
    ValueTeacherType value;
};

#endif //MIACIS_NEURAL_NETWORK_HPP