#ifndef MIACIS_NEURAL_NETWORK_HPP
#define MIACIS_NEURAL_NETWORK_HPP

#include"position.hpp"
#include<vector>
#include<primitiv/primitiv.h>

//primitiv
using namespace primitiv;
namespace F = primitiv::functions;
namespace I = primitiv::initializers;
namespace O = primitiv::optimizers;

//出力のチャンネル数:各マスに対する移動元方向(10) * 2 + 持ち駒7
constexpr uint32_t POLICY_CHANNEL_NUM = 27;

//評価パラメータを読み書きするデフォルトのファイル名
#ifdef USE_CATEGORICAL
const std::string MODEL_PATH = "cv.model";
#else
const std::string MODEL_PATH = "sv.model";
#endif

//型のエイリアス
using CalcType = float;
//using PolicyType = std::array<float, SQUARE_NUM * POLICY_CHANNEL_NUM>;
using PolicyType = std::vector<float>;
#ifdef USE_CATEGORICAL
constexpr int32_t BIN_SIZE = 51;
constexpr double VALUE_WIDTH = (MAX_SCORE - MIN_SCORE) / BIN_SIZE;
using ValueType = std::array<float, BIN_SIZE>;
using ValueTeacher = uint32_t;
#else
constexpr int32_t BIN_SIZE = 1;
using ValueType = float;
using ValueTeacher = float;
#endif

inline ValueType reverse(ValueType value) {
#ifdef USE_CATEGORICAL
    std::reverse(value.begin(), value.end());
    return value;
#else
#ifdef USE_SIGMOID
    return 1.0f - value;
#else
    return -value;
#endif
#endif
    //カテゴリカルなら反転を返す
}

struct TeacherType {
    uint32_t policy;
    ValueTeacher value;
};


constexpr int32_t BLOCK_NUM = 20;
constexpr int32_t KERNEL_SIZE = 3;
constexpr int32_t CHANNEL_NUM = 64;
constexpr int32_t VALUE_HIDDEN_NUM = 256;

template <typename Var>
class NeuralNetwork : public primitiv::Model {
public:
    NeuralNetwork() {
        add("first_filter", first_filter);
        for (int32_t i = 0; i < BLOCK_NUM; i++) {
            for (int32_t j = 0; j < 2; j++) {
                add("filter" + std::to_string(i) + std::to_string(j), filter[i][j]);
            }
        }
        add("value_filter", value_filter);
        add("value_pw_fc1", value_pw_fc1);
        add("value_pb_fc1", value_pb_fc1);
        add("value_pw_fc2", value_pw_fc2);
        add("value_pb_fc2", value_pb_fc2);
        add("policy_filter", policy_filter);
    }

    void init() {
        first_filter.init({KERNEL_SIZE, KERNEL_SIZE, INPUT_CHANNEL_NUM, CHANNEL_NUM}, I::XavierUniformConv2D());
        for (int32_t i = 0; i < BLOCK_NUM; i++) {
            for (int32_t j = 0; j < 2; j++) {
                filter[i][j].init({KERNEL_SIZE, KERNEL_SIZE, CHANNEL_NUM, CHANNEL_NUM}, I::XavierUniformConv2D());
            }
        }
        value_filter.init({1, 1, CHANNEL_NUM, CHANNEL_NUM}, I::XavierUniformConv2D());
        value_pw_fc1.init({VALUE_HIDDEN_NUM, SQUARE_NUM * CHANNEL_NUM}, I::XavierUniform());
        value_pb_fc1.init({VALUE_HIDDEN_NUM}, I::Constant(0));
        value_pw_fc2.init({BIN_SIZE, VALUE_HIDDEN_NUM}, I::XavierUniform());
        value_pb_fc2.init({BIN_SIZE}, I::Constant(0));
        policy_filter.init({1, 1, CHANNEL_NUM, POLICY_CHANNEL_NUM}, I::XavierUniformConv2D());
    }

    std::pair<Var, Var> feedForward(std::vector<float>& input) {
        uint32_t batch_size = (uint32_t)(input.size() / (SQUARE_NUM * INPUT_CHANNEL_NUM));
        Var x = F::input<Var>(Shape({9, 9, INPUT_CHANNEL_NUM}, batch_size), input);

        x = F::conv2d(x, F::parameter<Var>(first_filter), 1, 1, 1, 1, 1, 1);
        x = normalize(x);
        x = F::relu(x);

        for (int32_t i = 0; i < BLOCK_NUM; i++) {
            Var t = x;

            x = F::conv2d(x, F::parameter<Var>(filter[i][0]), 1, 1, 1, 1, 1, 1);
            x = normalize(x);
            x = F::relu(x);

            x = F::conv2d(x, F::parameter<Var>(filter[i][1]), 1, 1, 1, 1, 1, 1);
            x = normalize(x);
            x = F::relu(x + t);
        }

        //ここから分岐
        //policy
        Var policy = F::conv2d(x, F::parameter<Var>(policy_filter), 0, 0, 1, 1, 1, 1);

        //value
        Var value = F::conv2d(x, F::parameter<Var>(value_filter), 0, 0, 1, 1, 1, 1);
        value = normalize(value);
        value = F::relu(value);
        Var value_w_fc1 = F::parameter<Var>(value_pw_fc1);
        Var value_b_fc1 = F::parameter<Var>(value_pb_fc1);
        value = F::relu(F::matmul(value_w_fc1, F::flatten(value)) + value_b_fc1);
        Var value_w_fc2 = F::parameter<Var>(value_pw_fc2);
        Var value_b_fc2 = F::parameter<Var>(value_pb_fc2);
        value = F::matmul(value_w_fc2, value) + value_b_fc2;

#ifdef USE_CATEGORICAL
#else
#ifdef USE_SIGMOID
        value = F::sigmoid(value);
#else
        value = F::tanh(value);
#endif
#endif

        return { policy, value };
    }

    std::pair<PolicyType, ValueType> policyAndValue(const Position& pos) {
        std::vector<float> input = pos.makeFeature();
        auto y = feedForward(input);
        auto policy = F::flatten(y.first);

#ifdef USE_CATEGORICAL
        auto value = F::softmax(y.second, 0).to_vector();
        ValueType retval;
        std::copy(value.begin(), value.end(), retval.begin());
        return { policy.to_vector(), retval };
#else
        return { policy.to_vector(), y.second.to_float() };
#endif
    }

    std::pair<std::vector<PolicyType>, std::vector<ValueType>> policyAndValueBatch(std::vector<float> inputs) {
        auto y = feedForward(inputs);

        auto batch_size = inputs.size() / (SQUARE_NUM * INPUT_CHANNEL_NUM);

        std::vector<PolicyType> policies(batch_size);
        std::vector<ValueType> values(batch_size);

        auto policy = F::flatten(y.first).to_vector();
        for (int32_t i = 0; i < batch_size; i++) {
            policies[i].resize(POLICY_CHANNEL_NUM * SQUARE_NUM);
            for (int32_t j = 0; j < POLICY_CHANNEL_NUM * SQUARE_NUM; j++) {
                policies[i][j] = policy[i * POLICY_CHANNEL_NUM * SQUARE_NUM + j];
            }
        }

#ifdef USE_CATEGORICAL
        auto value = F::softmax(y.second, 0).to_vector();
        for (int32_t i = 0; i < batch_size; i++) {
            for (int32_t j = 0; j < BIN_SIZE; j++) {
                values[i][j] = value[i * BIN_SIZE + j];
            }
        }
#else
        auto value = y.second.to_vector();
        for (int32_t i = 0; i < batch_size; i++) {
            values[i] = value[i];
        }
#endif
        return { policies, values };
    }

#ifdef USE_CATEGORICAL
    std::pair<Var, Var> loss(std::vector<float>& input, std::vector<uint32_t>& policy_labels, std::vector<uint32_t >& value_labels) {
#else
    std::pair<Var, Var> loss(std::vector<float>& input, std::vector<uint32_t>& policy_labels, std::vector<float>& value_teachers) {
#endif
        auto y = feedForward(input);

        auto logits = F::flatten(y.first);

        Var policy_loss = F::softmax_cross_entropy(logits, policy_labels, 0);

#ifdef USE_CATEGORICAL
        Var value_loss = F::softmax_cross_entropy(y.second, value_labels, 0);
#else
        const Var value_t = F::input<Var>(Shape({1}, (uint32_t)value_teachers.size()), value_teachers);
#ifdef USE_SIGMOID
        Var value_loss = -value_t * F::log(y.second) -(1 - value_t) * F::log(1 - y.second);
#else
        Var value_loss = (y.second - value_t) * (y.second - value_t);
#endif
#endif
        return { F::batch::mean(policy_loss), F::batch::mean(value_loss) };
    }

    Var normalize(Var& x) {
        return F::batch::normalize(x);
        static constexpr int32_t G = 32;
        assert(CHANNEL_NUM % G == 0);
        auto batch_size = x.shape().batch();
        x = F::reshape(x, Shape({SQUARE_NUM * CHANNEL_NUM / G, G}, batch_size));
        Var m = F::mean(x, 0);
        m = F::broadcast(m, 0, SQUARE_NUM * CHANNEL_NUM / G);
        Var s = F::mean((x - m) * (x - m), 0);
        s = F::broadcast(s, 0, SQUARE_NUM * CHANNEL_NUM / G);
        x = (x - m) / F::sqrt(s + 1e-5);
        x = F::reshape(x, Shape({9, 9, CHANNEL_NUM}, batch_size));

        return x;
    }
    
private:

    Parameter first_filter;
    Parameter filter[BLOCK_NUM][2];
    Parameter value_filter;
    Parameter value_pw_fc1;
    Parameter value_pb_fc1;
    Parameter value_pw_fc2;
    Parameter value_pb_fc2;
    Parameter policy_filter;
};

extern std::unique_ptr<NeuralNetwork<Tensor>> nn;

#endif //MIACIS_NEURAL_NETWORK_HPP
