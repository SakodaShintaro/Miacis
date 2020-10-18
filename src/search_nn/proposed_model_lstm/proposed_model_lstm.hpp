#ifndef MIACIS_PROPOSED_MODEL_LSTM_HPP
#define MIACIS_PROPOSED_MODEL_LSTM_HPP

#include "../base_model/base_model.hpp"

class ProposedModelLSTMImpl : public BaseModel {
public:
    ProposedModelLSTMImpl() : ProposedModelLSTMImpl(SearchOptions()) {}
    explicit ProposedModelLSTMImpl(SearchOptions search_options);

    //root局面について探索を行って一番良い指し手を返す関数
    Move think(Position& root, int64_t time_limit) override;

    //ミニバッチデータに対して損失を計算する関数(現在のところバッチサイズは1のみに対応)
    std::vector<torch::Tensor> loss(const std::vector<LearningData>& data) override;

    //インタンスから下のクラス変数を参照するための関数
    std::string modelPrefix() override { return "proposed_model_lstm"; }

private:
    //探索
    std::vector<torch::Tensor> search(std::vector<Position>& positions);

    //各部分の推論
    torch::Tensor embed(const std::vector<Position>& positions);
    torch::Tensor simulationPolicy(const torch::Tensor& x);
    torch::Tensor readoutPolicy(const torch::Tensor& x);

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