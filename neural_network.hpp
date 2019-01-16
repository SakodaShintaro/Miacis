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
#ifdef USE_SIGMOID
    return 1.0f - value;
#else
    return -value;
#endif
    //カテゴリカルなら反転を返す
}

using TeacherType = std::pair<uint32_t, ValueType>;

//#define PRINT

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
        value_pw_fc2.init({1, VALUE_HIDDEN_NUM}, I::XavierUniform());
        value_pb_fc2.init({1}, I::Constant(0));
        policy_filter.init({1, 1, CHANNEL_NUM, POLICY_CHANNEL_NUM}, I::XavierUniformConv2D());
    }

    std::pair<Var, Var> feedForward(std::vector<float>& input, uint32_t batch_size) {
        Var x = F::input<Var>(Shape({9, 9, INPUT_CHANNEL_NUM}, batch_size), input);

        //conv
        Var first_filter_var = F::parameter<Var>(first_filter);
        x = F::conv2d(x, first_filter_var, 1, 1, 1, 1, 1, 1);

        //Normalize
        x = normalize(x);

        //ReLU
        x = F::relu(x);

        for (int32_t i = 0; i < BLOCK_NUM; i++) {
            Var t = x;

            //conv0
            Var filter0 = F::parameter<Var>(filter[i][0]);
            x = F::conv2d(x, filter0, 1, 1, 1, 1, 1, 1);

            //Batch Norm
            x = normalize(x);

            //ReLU
            x = F::relu(x);

            //conv1
            Var filter1 = F::parameter<Var>(filter[i][1]);
            x = F::conv2d(x, filter1, 1, 1, 1, 1, 1, 1);

            //Batch Norm
            x = normalize(x);

            //Residual
            x = F::relu(x + t);
        }

        //ここから分岐.まずvalueに1×1convを適用
        Var conv_value_filter = F::parameter<Var>(value_filter);
        Var value = F::conv2d(x, conv_value_filter, 0, 0, 1, 1, 1, 1);

        //Batch Norm
        value = normalize(value);

        //ReLU
        value = F::relu(value);

        //全結合1
        Var value_w_fc1 = F::parameter<Var>(value_pw_fc1);
        Var value_b_fc1 = F::parameter<Var>(value_pb_fc1);
        value = F::relu(F::matmul(value_w_fc1, F::flatten(value)) + value_b_fc1);

        //全結合2
        Var value_w_fc2 = F::parameter<Var>(value_pw_fc2);
        Var value_b_fc2 = F::parameter<Var>(value_pb_fc2);
#ifdef USE_SIGMOID
        value_fc2 = F::sigmoid(F::matmul(value_w_fc2, value_fc1) + value_b_fc2);
#else
//        for (auto& e : F::matmul(value_w_fc2, value).to_vector()) {
//            std::cout << e << " ";
//        }
//        std::cout << "b =  " << value_b_fc2.to_float() << std::endl;
        value = F::tanh(F::matmul(value_w_fc2, value) + value_b_fc2);
#endif

        //policyの1×1conv
        Var conv_policy_filter = F::parameter<Var>(policy_filter);
        Var policy = F::conv2d(x, conv_policy_filter, 0, 0, 1, 1, 1, 1);

        return { policy, value };
    }

    std::pair<PolicyType, ValueType> policyAndValue(const Position& pos) {
        std::vector<float> input = pos.makeFeature();
        Graph g;
        Graph::set_default(g);
        auto y = feedForward(input, 1);
        auto policy = F::flatten(y.first);
        auto value = y.second;
        return { policy.to_vector(), value.to_float() };
    }

    std::pair<Var, Var> loss(std::vector<float>& input, std::vector<uint32_t>& labels, std::vector<float>& value_teachers, uint32_t batch_size) {
        auto y = feedForward(input, batch_size);
        const Var value_t  = F::input<Var>(Shape({1}, batch_size), value_teachers);

        auto logits = F::flatten(y.first);

        Var policy_loss = F::softmax_cross_entropy(logits, labels, 0);
#ifdef USE_SIGMOID
        Var value_loss = -value_t * F::log(y.second) -(1 - value_t) * F::log(1 - y.second);
#else
        Var value_loss = (y.second - value_t) * (y.second - value_t);
//        for (const auto& e : y.second.to_vector()) {
//            std::cout << e << " ";
//        }
//        std::cout << std::endl;
//        for (const auto& e : value_t.to_vector()) {
//            std::cout << e << " ";
//        }
//        std::cout << std::endl;
#endif

        return { F::batch::mean(policy_loss), F::batch::mean(value_loss) };
    }

    Var normalize(Var& x) {
        //パラメータを変換
//        Var gamma_var = F::parameter<Var>(gamma);
//        Var beta_var  = F::parameter<Var>(beta);

        static constexpr int32_t G = 32;
        assert(CHANNEL_NUM % G == 0);
        auto batch_size = x.shape().batch();
        x = F::reshape(x, Shape({SQUARE_NUM * CHANNEL_NUM / G, G}, batch_size));
        Var m = F::mean(x, 0);
        m = F::broadcast(m, 0, SQUARE_NUM * CHANNEL_NUM / G);
        Var s = F::mean((x - m) * (x - m), 0);
        s = F::broadcast(s, 0, SQUARE_NUM * CHANNEL_NUM / G);
        x = (x - m) / F::sqrt(s + 1e-5);
//        x = gamma * m + beta;
        x = F::reshape(x, Shape({9, 9, CHANNEL_NUM}, batch_size));

        return x;
    }
    
    void print() {
        auto p1 = F::parameter<Var>(policy_filter);
        auto vp = F::flatten(p1).to_vector();
        for (int32_t i = 0; i < 100; i++) {
            std::cout << vp[i] << " \n"[i == 99];
        }

        auto w1 = F::parameter<Var>(value_pw_fc1);
        auto vw1 = F::flatten(w1).to_vector();
//        for (const auto& e : vw1) {
//            std::cout << e << " ";
//        }
//        std::cout << std::endl;

        auto b1 = F::parameter<Var>(value_pb_fc1);
        auto vb1 = F::flatten(b1).to_vector();
        for (const auto& e : vb1) {
            std::cout << e << " ";
        }
        std::cout << std::endl;

        auto w2 = F::parameter<Var>(value_pw_fc2);
        auto vw2 = F::flatten(w2).to_vector();
        for (const auto& e : vw2) {
            std::cout << e << " ";
        }
        std::cout << std::endl;

        auto b2 = F::parameter<Var>(value_pb_fc2);
        auto vb2 = F::flatten(b2).to_vector();
        for (const auto& e : vb2) {
            std::cout << e << " ";
        }
        std::cout << std::endl;
    }
private:
    static constexpr int32_t BLOCK_NUM = 2;
    static constexpr int32_t KERNEL_SIZE = 3;
    static constexpr int32_t CHANNEL_NUM = 32;
    static constexpr int32_t VALUE_HIDDEN_NUM = 256;

    Parameter first_filter;
    Parameter first_gamma, first_beta;
    Parameter filter[BLOCK_NUM][2];
    Parameter gamma[BLOCK_NUM][2], beta[BLOCK_NUM][2];
    Parameter value_filter;
    Parameter value_gamma, value_beta;
    Parameter value_pw_fc1;
    Parameter value_pb_fc1;
    Parameter value_pw_fc2;
    Parameter value_pb_fc2;
    Parameter policy_filter;
};

extern std::unique_ptr<NeuralNetwork<Tensor>> nn;

#endif //MIACIS_NEURAL_NETWORK_HPP