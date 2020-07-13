#ifndef MIACIS_PROPOSED_LSTM_HPP
#define MIACIS_PROPOSED_LSTM_HPP

#include<torch/torch.h>

//提案手法で用いるNN
//入力:局面の分散表現の系列
//出力:一番最後の局面における方策、状態価値

class ProposedModel : public torch::nn::Module {
public:
    ProposedModel();
    std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& x);
    torch::Tensor loss(const torch::Tensor& x, const torch::Tensor& t);
    void resetState();
private:
    torch::nn::LSTM lstm_{ nullptr };
    torch::nn::Linear policy_head_{ nullptr };
    torch::nn::Linear value_head_{ nullptr };
    torch::Tensor h_;
    torch::Tensor c_;
};

#endif //MIACIS_PROPOSED_LSTM_HPP