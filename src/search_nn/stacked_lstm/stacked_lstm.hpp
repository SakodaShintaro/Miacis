#ifndef MIACIS_STACKED_LSTM_HPP
#define MIACIS_STACKED_LSTM_HPP

#include "../state_encoder.hpp"

class StackedLSTMImpl : public torch::nn::Module {
public:
    StackedLSTMImpl() : StackedLSTMImpl(SearchOptions()) {}
    explicit StackedLSTMImpl(SearchOptions search_options);

    //root局面について探索を行って一番良い指し手を返す関数
    Move think(Position& root, int64_t time_limit);

    //ミニバッチデータに対して損失を計算する関数(現在のところバッチサイズは1のみに対応)
    std::vector<torch::Tensor> loss(const std::vector<LearningData>& data, bool use_policy_gradient);
    std::vector<torch::Tensor> validationLoss(const std::vector<LearningData>& data) { return loss(data, true); }

    //GPUにネットワークを送る関数
    void setGPU(int16_t gpu_id, bool fp16 = false);

    //事前学習したモデルを読み込む関数
    void loadPretrain(const std::string& encoder_path, const std::string& policy_head_path, const std::string& value_head_path);

    //インタンスから下のクラス変数を参照するための関数
    static std::string modelPrefix() { return MODEL_PREFIX; }
    static std::string defaultModelName() { return DEFAULT_MODEL_NAME; }

    //学習の設定を定める関数
    void setOption(bool freeze_encoder, float gamma);

private:
    //評価パラメータを読み書きするファイルのprefix
    static const std::string MODEL_PREFIX;

    //デフォルトで読み書きするファイル名
    static const std::string DEFAULT_MODEL_NAME;

    //各部分の推論
    torch::Tensor embed(const std::vector<float>& inputs);
    torch::Tensor simulationPolicy(const torch::Tensor& x);
    torch::Tensor readoutPolicy(const torch::Tensor& x);
    torch::Tensor predictNextState(const torch::Tensor& pre_state, const torch::Tensor& abstract_action);

    //探索全体
    std::vector<torch::Tensor> search(const std::vector<float>& inputs);

    //探索に関するオプション
    SearchOptions search_options_;

    //---------------
    //    Encoder
    //---------------
    StateEncoder encoder_{ nullptr };

    //------------------
    //    Value Head
    //------------------
    torch::nn::Linear value_head_{ nullptr };

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
    torch::Tensor readout_h_;
    torch::Tensor readout_c_;

    //デバイスとfp16化
    torch::Device device_;
    bool fp16_;

    //エンコーダを固定して学習するかどうか
    bool freeze_encoder_;
};
TORCH_MODULE(StackedLSTM);

#endif //MIACIS_STACKED_LSTM_HPP