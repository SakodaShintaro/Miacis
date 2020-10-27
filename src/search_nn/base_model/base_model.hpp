#ifndef MIACIS_BASE_MODEL_HPP
#define MIACIS_BASE_MODEL_HPP

#include "../state_encoder.hpp"
#include <torch/torch.h>

class BaseModel : public torch::nn::Module {
public:
    explicit BaseModel(SearchOptions options);

    //root局面について探索を行って一番良い指し手を返す関数
    virtual Move think(Position& root, int64_t time_limit);

    //ミニバッチデータに対して損失を計算する関数(現在のところバッチサイズは1のみに対応)
    std::vector<torch::Tensor> loss(const std::vector<LearningData>& data);

    //モデル名に関する関数
    virtual std::string modelPrefix() = 0;
    std::string defaultModelName() { return modelPrefix() + ".model"; };

    //GPUにネットワークを送る関数
    void setGPU(int16_t gpu_id, bool fp16 = false);

    //事前学習したモデルを読み込む関数
    void loadPretrain(const std::string& encoder_path, const std::string& policy_head_path, const std::string& value_head_path);

    //学習の設定を定める関数
    void setOption(bool freeze_encoder);

protected:
    //探索
    virtual std::vector<torch::Tensor> search(std::vector<Position>& positions) = 0;

    //局面のバッチを埋め込む関数
    torch::Tensor embed(const std::vector<Position>& positions);

    //探索に関するオプション
    SearchOptions search_options_;

    //Simulation用のネットワーク
    StateEncoder encoder_{ nullptr };
    torch::nn::Linear sim_policy_head_{ nullptr };
    torch::nn::Linear base_value_head_{ nullptr };

    //デバイスとfp16化
    torch::Device device_;
    bool fp16_;

    //エンコーダを固定して学習するかどうか
    bool freeze_encoder_;
};

#endif //MIACIS_BASE_MODEL_HPP