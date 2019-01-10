#ifndef MIACIS_NEURAL_NETWORK_HPP
#define MIACIS_NEURAL_NETWORK_HPP

#include"position.hpp"
#include<primitiv/primitiv.h>

//primitiv
using namespace primitiv;
namespace F = primitiv::functions;
namespace I = primitiv::initializers;
namespace O = primitiv::optimizers;

//出力のチャンネル数:各マスに対する移動元方向(10) * 2 + 持ち駒7
constexpr uint32_t POLICY_CHANNEL_NUM = 27;

//基本的に読み書きするパス
const std::string MODEL_PATH = "cnn.model";

//型のエイリアス
using CalcType = float;
//using PolicyType = std::array<float, SQUARE_NUM * POLICY_CHANNEL_NUM>;
using PolicyType = std::vector<float>;
using ValueType = float;

inline ValueType reverse(ValueType value) {
    return -value;
    //カテゴリカルなら反転を返す
}

using TeacherType = std::pair<PolicyType, ValueType>;

template <typename Var>
class NeuralNetwork : public primitiv::Model {
public:
    NeuralNetwork() {
        for (int32_t i = 0; i < LAYER_NUM; i++) {
            add("filter" + std::to_string(i), filter[i]);
        }
        add("value_filter", value_filter);

        add("value_pw_fc1", value_pw_fc1);

        add("value_pb_fc1", value_pb_fc1);

        add("value_pw_fc2", value_pw_fc2);

        add("value_pb_fc2", value_pb_fc2);

        add("policy_filter", policy_filter);
    }

    void init() {
        for (int32_t i = 0; i < LAYER_NUM; i++) {
            filter[i].init({KERNEL_SIZE, KERNEL_SIZE, (i == 0 ? INPUT_CHANNEL_NUM : CHANNEL_NUM), CHANNEL_NUM}
                    , I::XavierUniformConv2D());
        }
        value_filter.init({1, 1, CHANNEL_NUM, CHANNEL_NUM}, I::XavierUniformConv2D());

        value_pw_fc1.init({VALUE_HIDDEN_NUM, SQUARE_NUM * CHANNEL_NUM}, I::XavierUniform());

        value_pb_fc1.init({VALUE_HIDDEN_NUM}, I::Constant(0));

        value_pw_fc2.init({1, VALUE_HIDDEN_NUM}, I::XavierUniform());

        value_pb_fc2.init({1}, I::Constant(0));

        policy_filter.init({1, 1, CHANNEL_NUM, POLICY_CHANNEL_NUM}, I::XavierUniformConv2D());
    }

    std::pair<Var, Var> feedForward(std::vector<float>& input, uint32_t batch_size) {
        const Var x = F::input<Var>(Shape({9, 9, INPUT_CHANNEL_NUM}, batch_size), input);
        Var conv_filter[LAYER_NUM];
        for (int32_t i = 0; i < LAYER_NUM; i++) {
            conv_filter[i] = F::parameter<Var>(filter[i]);
            conv[i] = F::relu(F::conv2d((i == 0 ? x : conv[i - 1]), conv_filter[i],
                    1, 1, 1, 1, 1, 1));
        }
        Var conv_value_filter = F::parameter<Var>(value_filter);
        value_conv = F::relu(F::conv2d(conv[LAYER_NUM - 1], conv_value_filter, 0, 0, 1, 1, 1, 1));

        Var value_w_fc1 = F::parameter<Var>(value_pw_fc1);
        Var value_b_fc1 = F::parameter<Var>(value_pb_fc1);
        value_fc1 = F::relu(F::matmul(value_w_fc1, F::flatten(value_conv)) + value_b_fc1);

        Var value_w_fc2 = F::parameter<Var>(value_pw_fc2);
        Var value_b_fc2 = F::parameter<Var>(value_pb_fc2);
        value_fc2 = F::tanh(F::matmul(value_w_fc2, value_fc1) + value_b_fc2);

        Var conv_policy_filter = F::parameter<Var>(policy_filter);
        policy_conv = F::conv2d(conv[LAYER_NUM - 1], conv_policy_filter, 0, 0, 1, 1, 1, 1);

        return { policy_conv, value_fc2 };
    }

    std::pair<PolicyType, ValueType> policyAndValue(const Position& pos) {
        std::vector<float> input = pos.makeFeature();
        auto y = feedForward(input, 1);
        auto policy = y.first;
        auto value = y.second;
        return { policy.to_vector(), value.to_float() };
    }

    std::pair<Var, Var> loss(std::vector<float>& input, std::vector<float>& policy_teachers, std::vector<float>& value_teachers, uint32_t batch_size) {
        auto y = feedForward(input, batch_size);
        const Var policy_t = F::input<Var>(Shape({9 * 9 * POLICY_CHANNEL_NUM}, batch_size), policy_teachers);
        const Var value_t = F::input<Var>(Shape({1}, batch_size), value_teachers);

        auto logits = F::flatten(y.first);
        std::cout << logits.shape().to_string() << std::endl;
        std::cout << policy_t.shape().to_string() << std::endl;

        Var policy_loss = F::softmax_cross_entropy(logits, policy_t, 0);
        Var value_loss = F::pow(y.second - value_t, 2.0);

        return { F::batch::mean(policy_loss), F::batch::mean(value_loss) };
    }

    std::pair<Var, Var> loss(std::vector<float>& input, std::vector<uint32_t>& labels, std::vector<float>& value_teachers, uint32_t batch_size) {
        auto y = feedForward(input, batch_size);
        const Var value_t  = F::input<Var>(Shape({1}, batch_size), value_teachers);

        auto logits = F::flatten(y.first);

        Var policy_loss = F::softmax_cross_entropy(logits, labels, 0);
        Var value_loss = F::pow(y.second - value_t, 2.0);

        return { F::batch::mean(policy_loss), F::batch::mean(value_loss) };
    }
private:
    static constexpr int32_t LAYER_NUM = 3;
    static constexpr int32_t KERNEL_SIZE = 3;
    static constexpr int32_t CHANNEL_NUM = 16;
    static constexpr int32_t VALUE_HIDDEN_NUM = 256;
    Var conv[LAYER_NUM];
    Var value_conv;
    Var value_fc1;
    Var value_fc2;
    Var policy_conv;

    Parameter filter[LAYER_NUM];
    Parameter value_filter;
    Parameter value_pw_fc1;
    Parameter value_pb_fc1;
    Parameter value_pw_fc2;
    Parameter value_pb_fc2;
    Parameter policy_filter;
};

extern std::unique_ptr<NeuralNetwork<Tensor>> nn;

#endif //MIACIS_NEURAL_NETWORK_HPP