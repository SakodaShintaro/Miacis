#ifndef MIACIS_PROPOSED_MODEL_HPP
#define MIACIS_PROPOSED_MODEL_HPP

#include "../base_model/base_model.hpp"

class ProposedModelImpl : public BaseModel {
public:
    ProposedModelImpl() : ProposedModelImpl(SearchOptions()) {}
    explicit ProposedModelImpl(SearchOptions search_options);

    //root局面について探索を行って一番良い指し手を返す関数
    Move think(Position& root, int64_t time_limit) override;

    //ミニバッチデータに対して損失を計算する関数(現在のところバッチサイズは1のみに対応)
    std::vector<torch::Tensor> loss(const std::vector<LearningData>& data, bool use_policy_gradient) override;
    std::vector<torch::Tensor> validationLoss(const std::vector<LearningData>& data) { return loss(data, true); }

    //インタンスから下のクラス変数を参照するための関数
    std::string modelPrefix() override { return MODEL_PREFIX; }
    std::string defaultModelName() override { return DEFAULT_MODEL_NAME; }

private:
    //評価パラメータを読み書きするファイルのprefix
    static const std::string MODEL_PREFIX;

    //デフォルトで読み書きするファイル名
    static const std::string DEFAULT_MODEL_NAME;

    //探索
    std::vector<torch::Tensor> search(std::vector<Position>& positions);

    //各部分の推論
    torch::Tensor embed(const std::vector<float>& inputs);
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
TORCH_MODULE(ProposedModel);

#endif //MIACIS_PROPOSED_MODEL_HPP