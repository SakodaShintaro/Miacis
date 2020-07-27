#ifndef MIACIS_MUZERO_HPP
#define MIACIS_MUZERO_HPP

#include "../state_encoder.hpp"

class MuZeroImpl : public torch::nn::Module {
public:
    MuZeroImpl() : MuZeroImpl(SearchOptions()) {}
    explicit MuZeroImpl(SearchOptions search_options);

    //root局面について探索を行って一番良い指し手を返す関数
    Move think(Position& root, int64_t time_limit);

    //ミニバッチデータに対して損失を計算する関数(現在のところバッチサイズは1のみに対応)
    std::vector<torch::Tensor> loss(const std::vector<LearningData>& data);

    //GPUにネットワークを送る関数
    void setGPU(int16_t gpu_id, bool fp16 = false);

    //インタンスから下のクラス変数を参照するための関数
    static std::string modelPrefix() { return MODEL_PREFIX; }
    static std::string defaultModelName() { return DEFAULT_MODEL_NAME; }

private:
    //評価パラメータを読み書きするファイルのprefix
    static const std::string MODEL_PREFIX;

    //デフォルトで読み書きするファイル名
    static const std::string DEFAULT_MODEL_NAME;

    //各部分の推論
    torch::Tensor embed(const std::vector<float>& inputs);
    torch::Tensor inferPolicy(const Position& pos);

    //探索に関するオプション
    SearchOptions search_options_;

    //---------------
    //    Encoder
    //---------------
    StateEncoder encoder_{ nullptr };

    //---------------
    //    Decoder
    //---------------
    torch::nn::Linear policy_{ nullptr };
    torch::nn::Linear value_{ nullptr };

    //-----------------
    //    環境モデル
    //-----------------
    StateEncoder env_model_{ nullptr };

    //デバイスとfp16化
    torch::Device device_;
    bool fp16_;
};
TORCH_MODULE(MuZero);

#endif //MIACIS_MUZERO_HPP