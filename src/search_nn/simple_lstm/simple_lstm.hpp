#ifndef MIACIS_SIMPLE_LSTM_HPP
#define MIACIS_SIMPLE_LSTM_HPP

#include "../base_model/base_model.hpp"

class SimpleLSTMImpl : public BaseModel {
public:
    SimpleLSTMImpl() : SimpleLSTMImpl(SearchOptions()) {}
    explicit SimpleLSTMImpl(const SearchOptions& search_options);

    //インタンスから下のクラス変数を参照するための関数
    std::string modelPrefix() override { return "simple_lstm"; }

private:
    //探索
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> search(std::vector<Position>& positions) override;

    //各部分の推論
    std::tuple<torch::Tensor, torch::Tensor> readout(const torch::Tensor& x);

    torch::nn::LSTM readout_lstm_{ nullptr };
    torch::nn::Linear readout_policy_head_{ nullptr };
    torch::nn::Linear readout_value_head_{ nullptr };
    torch::Tensor readout_h_;
    torch::Tensor readout_c_;
};
TORCH_MODULE(SimpleLSTM);

#endif //MIACIS_SIMPLE_LSTM_HPP
