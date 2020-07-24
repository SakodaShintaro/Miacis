#ifndef MIACIS_MCTS_NET_HPP
#define MIACIS_MCTS_NET_HPP

#include "hash_table_for_mcts_net.hpp"
#include "../state_encoder.hpp"

class MCTSNetImpl : public torch::nn::Module {
public:
    MCTSNetImpl() : MCTSNetImpl(SearchOptions()) {}
    explicit MCTSNetImpl(const SearchOptions& search_options);

    //root局面について探索を行って一番良い指し手を返す関数
    Move think(Position& root, int64_t time_limit, bool save_info_to_learn = false);

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
    torch::Tensor simulationPolicy(const torch::Tensor& h);
    torch::Tensor embed(const std::vector<float>& inputs);
    torch::Tensor backup(const torch::Tensor& h1, const torch::Tensor& h2);
    torch::Tensor readoutPolicy(const torch::Tensor& h);

    //探索に関するオプション
    SearchOptions search_options_;

    //置換表は1個
    HashTableForMCTSNet hash_table_;

    //使用するニューラルネットワーク
    //simulation policy network
    torch::nn::Linear simulation_policy_{ nullptr };

    //embed network
    StateEncoder encoder_{ nullptr };

    //backup network
    torch::nn::Linear backup_update_{ nullptr };
    torch::nn::Linear backup_gate_{ nullptr };

    //readout network
    torch::nn::Linear readout_policy_{ nullptr };

    //デバイスとfp16化
    torch::Device device_;
    bool fp16_;

    //学習で使うために探索中に保存しておく値
    std::vector<torch::Tensor> root_h_;
    std::vector<torch::Tensor> probs_;
};
TORCH_MODULE(MCTSNet);

#endif //MIACIS_MCTS_NET_HPP