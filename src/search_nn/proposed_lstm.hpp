#ifndef MIACIS_PROPOSED_LSTM_HPP
#define MIACIS_PROPOSED_LSTM_HPP

#include"../search_options.hpp"
#include"../include_switch.hpp"
#include<torch/torch.h>

//提案手法で用いるNN
//入力:局面の分散表現の系列
//出力:一番最後の局面における方策、状態価値

class ProposedModelImpl : public torch::nn::Module {
public:
    ProposedModelImpl() : ProposedModelImpl(SearchOptions()) {}
    explicit ProposedModelImpl(SearchOptions search_options);

    //探索を行って一番良い指し手を返す関数
    Move think(Position& root, int64_t time_limit, bool save_info_to_learn = false);

    torch::Tensor loss(const torch::Tensor& x, const torch::Tensor& t);
    void resetState();
private:
    torch::Tensor embed(const std::vector<float>& inputs);

    std::tuple<torch::Tensor, torch::Tensor> simulationPolicy(const torch::Tensor& x);
    std::tuple<torch::Tensor, torch::Tensor> readoutPolicy(const torch::Tensor& x);

    //探索に関するオプション
    SearchOptions search_options_;

    //---------------
    //    Encoder
    //---------------
    //最初にチャンネル数を変えるConv
    Conv2DwithBatchNorm first_conv_{ nullptr };

    //同じチャンネル数で残差ブロックを通す
    std::vector<ResidualBlock> blocks_;

    //最後にチャンネル数を絞るConv
    Conv2DwithBatchNorm last_conv_{ nullptr };

    //-------------------------
    //    Simulation Policy
    //-------------------------
    torch::nn::LSTM simulation_lstm_{ nullptr };
    torch::nn::Linear simulation_policy_head_{ nullptr };
    torch::nn::Linear simulation_value_head_{ nullptr };
    torch::Tensor simulation_h_;
    torch::Tensor simulation_c_;

    //----------------------
    //    Readout Policy
    //----------------------
    torch::nn::LSTM readout_lstm_{ nullptr };
    torch::nn::Linear readout_policy_head_{ nullptr };
    torch::nn::Linear readout_value_head_{ nullptr };
    torch::Tensor readout_h_;
    torch::Tensor readout_c_;

    //デバイスとfp16化
    torch::Device device_;
    bool fp16_;
};
TORCH_MODULE(ProposedModel);

#endif //MIACIS_PROPOSED_LSTM_HPP