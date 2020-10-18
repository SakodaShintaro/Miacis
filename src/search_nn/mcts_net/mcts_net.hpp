#ifndef MIACIS_MCTS_NET_HPP
#define MIACIS_MCTS_NET_HPP

#include "../base_model/base_model.hpp"
#include "hash_table_for_mcts_net.hpp"

class MCTSNetImpl : public BaseModel {
public:
    MCTSNetImpl() : MCTSNetImpl(SearchOptions()) {}
    explicit MCTSNetImpl(const SearchOptions& search_options);

    //root局面について探索を行って一番良い指し手を返す関数
    Move think(Position& root, int64_t time_limit) override;

    //ミニバッチデータに対して損失を計算する関数(現在のところバッチサイズは1のみに対応)
    std::vector<torch::Tensor> loss(const std::vector<LearningData>& data) override;

    //インタンスから下のクラス変数を参照するための関数
    std::string modelPrefix() override { return "mcts_net"; }

private:
    //各部分の推論
    torch::Tensor backup(const torch::Tensor& h1, const torch::Tensor& h2);

    //置換表は1個
    HashTableForMCTSNet hash_table_;

    //使用するニューラルネットワーク
    //backup network
    torch::nn::Linear backup_update_{ nullptr };
    torch::nn::Linear backup_gate_{ nullptr };

    //readout network
    torch::nn::Linear readout_policy_{ nullptr };
};
TORCH_MODULE(MCTSNet);

#endif //MIACIS_MCTS_NET_HPP