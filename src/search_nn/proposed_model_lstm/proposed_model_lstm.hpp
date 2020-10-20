#ifndef MIACIS_PROPOSED_MODEL_LSTM_HPP
#define MIACIS_PROPOSED_MODEL_LSTM_HPP

#include "../base_model/base_model.hpp"

class ProposedModelLSTMImpl : public BaseModel {
public:
    ProposedModelLSTMImpl() : ProposedModelLSTMImpl(SearchOptions()) {}
    explicit ProposedModelLSTMImpl(SearchOptions search_options);

    //インタンスから下のクラス変数を参照するための関数
    std::string modelPrefix() override { return "proposed_model_lstm"; }

private:
    //探索
    std::vector<torch::Tensor> search(std::vector<Position>& positions) override;

    //各部分の推論
    torch::Tensor readoutPolicy(const torch::Tensor& x, bool update_hidden_state);

    //----------------------
    //    Readout Policy
    //----------------------
    torch::nn::LSTM readout_lstm_{ nullptr };
    torch::nn::Linear readout_policy_head_{ nullptr };
    torch::Tensor readout_h_;
    torch::Tensor readout_c_;
};
TORCH_MODULE(ProposedModelLSTM);

#endif //MIACIS_PROPOSED_MODEL_LSTM_HPP