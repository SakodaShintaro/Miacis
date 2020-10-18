#ifndef MIACIS_PROPOSED_MODEL_TRANSFORMER_HPP
#define MIACIS_PROPOSED_MODEL_TRANSFORMER_HPP

#include "../api/include/modules/transformer.h"
#include "../base_model/base_model.hpp"

class ProposedModelTransformerImpl : public BaseModel {
public:
    ProposedModelTransformerImpl() : ProposedModelTransformerImpl(SearchOptions()) {}
    explicit ProposedModelTransformerImpl(const SearchOptions& search_options);

    //root局面について探索を行って一番良い指し手を返す関数
    Move think(Position& root, int64_t time_limit) override;

    //ミニバッチデータに対して損失を計算する関数(現在のところバッチサイズは1のみに対応)
    std::vector<torch::Tensor> loss(const std::vector<LearningData>& data) override;

    //インタンスから下のクラス変数を参照するための関数
    std::string modelPrefix() override { return "proposed_model_transformer"; }

private:
    //各部分の推論
    torch::Tensor embed(const std::vector<Position>& positions);
    torch::Tensor inferPolicy(const torch::Tensor& x, const std::vector<torch::Tensor>& history);
    torch::Tensor positionalEncoding(int64_t pos) const;

    //探索全体
    std::vector<torch::Tensor> search(std::vector<Position>& positions);

    //transformer
    torch::nn::Transformer transformer_{ nullptr };
    torch::nn::Linear policy_head_{ nullptr };
};
TORCH_MODULE(ProposedModelTransformer);

#endif //MIACIS_PROPOSED_MODEL_TRANSFORMER_HPP