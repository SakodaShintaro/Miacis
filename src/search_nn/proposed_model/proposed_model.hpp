#ifndef MIACIS_PROPOSED_MODEL_HPP
#define MIACIS_PROPOSED_MODEL_HPP

#include"../../search_options.hpp"
#include"../../include_switch.hpp"
#include<torch/torch.h>

class ProposedModelImpl : public torch::nn::Module {
public:
    ProposedModelImpl() : ProposedModelImpl(SearchOptions()) {}
    explicit ProposedModelImpl(SearchOptions search_options);

    //root局面について探索を行って一番良い指し手を返す関数
    Move think(Position& root, int64_t time_limit);

    //ミニバッチデータに対して損失を計算する関数(現在のところバッチサイズは1のみに対応)
    std::vector<torch::Tensor> loss(const std::vector<LearningData>& data);

    //GPUにネットワークを送る関数
    void setGPU(int16_t gpu_id, bool fp16 = false);

    //評価パラメータを読み書きするファイルのprefix
    static const std::string MODEL_PREFIX;

    //デフォルトで読み書きするファイル名
    static const std::string DEFAULT_MODEL_NAME;

    //インタンスから上記のクラス変数を参照するための関数
    static std::string modelPrefix() { return MODEL_PREFIX; }
    static std::string defaultModelName() { return DEFAULT_MODEL_NAME; }

private:
    //各部分の推論
    torch::Tensor embed(const std::vector<float>& inputs);
    std::tuple<torch::Tensor, torch::Tensor> simulationPolicy(const torch::Tensor& x);
    std::tuple<torch::Tensor, torch::Tensor> readoutPolicy(const torch::Tensor& x);

    //探索に関するオプション
    SearchOptions search_options_;

    //---------------
    //    Encoder
    //---------------
    Conv2DwithBatchNorm first_conv_{ nullptr };
    std::vector<ResidualBlock> blocks_;
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

    //出力方策の系列
    std::vector<torch::Tensor> outputs_;

    //デバイスとfp16化
    torch::Device device_;
    bool fp16_;
};
TORCH_MODULE(ProposedModel);

#endif //MIACIS_PROPOSED_MODEL_HPP