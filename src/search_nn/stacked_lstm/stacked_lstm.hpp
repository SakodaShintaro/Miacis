#ifndef MIACIS_STACKED_LSTM_HPP
#define MIACIS_STACKED_LSTM_HPP

#include "../base_model/base_model.hpp"

class StackedLSTMImpl : public BaseModel {
public:
    StackedLSTMImpl() : StackedLSTMImpl(SearchOptions()) {}
    explicit StackedLSTMImpl(const SearchOptions& search_options);

    //インタンスから下のクラス変数を参照するための関数
    std::string modelPrefix() override { return "stacked_lstm"; }

private:
    //探索
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> search(std::vector<Position>& positions) override;

    //各部分の推論
    torch::Tensor simulationPolicy(const torch::Tensor& x);
    std::tuple<torch::Tensor, torch::Tensor> readout(const torch::Tensor& x);
    torch::Tensor predictNextState(const torch::Tensor& pre_state, const torch::Tensor& abstract_action);

    //-------------------------
    //    Environment Model
    //-------------------------
    torch::nn::Linear env_model0_{ nullptr };
    torch::nn::Linear env_model1_{ nullptr };

    //-------------------------
    //    Simulation Policy
    //-------------------------
    torch::nn::LSTM simulation_lstm_{ nullptr };
    torch::nn::Linear simulation_policy_head_{ nullptr };
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
};
TORCH_MODULE(StackedLSTM);

#endif //MIACIS_STACKED_LSTM_HPP