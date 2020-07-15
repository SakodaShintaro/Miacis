#ifndef MIACIS_MCTS_NET_HPP
#define MIACIS_MCTS_NET_HPP

#include "hash_table_for_mcts_net.hpp"
#include "../search_options.hpp"

//MCTSを行うクラス
//想定の使い方は局面を放り投げて探索せよと投げることか
//なのでSearcherForPlayと置き換えられるように作れば良さそう
class MCTSNetImpl : public torch::nn::Module  {
public:
    MCTSNetImpl();
    explicit MCTSNetImpl(const SearchOptions& search_options);

    //探索を行って一番良い指し手を返す関数
    Move think(Position& root, int64_t time_limit, bool save_info_to_learn = false);

    //一つの局面について損失等を計算する関数
    torch::Tensor loss(const std::vector<LearningData>& data);

    //GPUにネットワークを送る関数
    void setGPU(int16_t gpu_id, bool fp16 = false);

    //評価パラメータを読み書きするファイルのprefix
    static const std::string MODEL_PREFIX;

    //デフォルトで読み書きするファイル名
    static const std::string DEFAULT_MODEL_NAME;

private:
    //各部分の推論
    torch::Tensor simulationPolicy(const torch::Tensor& h);
    torch::Tensor embed(const std::vector<float>& inputs);
    torch::Tensor embed(const torch::Tensor& x);
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
    //最初にチャンネル数を変えるConv
    Conv2DwithBatchNorm first_conv_{ nullptr };

    //同じチャンネル数で残差ブロックを通す
    std::vector<ResidualBlock> blocks_;

    //最後にまた絞るConv
    Conv2DwithBatchNorm last_conv_{ nullptr };

    //backup network
    torch::nn::Linear backup_linear_{ nullptr };

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