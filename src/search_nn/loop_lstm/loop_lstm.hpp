#ifndef MIACIS_LOOP_LSTM_HPP
#define MIACIS_LOOP_LSTM_HPP

#include "../base_model/base_model.hpp"

class LoopLSTMImpl : public BaseModel {
public:
    LoopLSTMImpl() : LoopLSTMImpl(SearchOptions()) {}
    explicit LoopLSTMImpl(const SearchOptions& search_options);

    //インタンスから下のクラス変数を参照するための関数
    std::string modelPrefix() override { return "loop_lstm"; }

private:
    //探索
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> search(std::vector<Position>& positions) override;

    torch::nn::LSTM readout_lstm_{ nullptr };
    torch::nn::Linear readout_policy_head_{ nullptr };
    torch::nn::Linear readout_value_head_{ nullptr };
    torch::Tensor readout_h_;
    torch::Tensor readout_c_;
};
TORCH_MODULE(LoopLSTM);

#endif //MIACIS_LOOP_LSTM_HPP