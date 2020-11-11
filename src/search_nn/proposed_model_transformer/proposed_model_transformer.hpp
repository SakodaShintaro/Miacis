#ifndef MIACIS_PROPOSED_MODEL_TRANSFORMER_HPP
#define MIACIS_PROPOSED_MODEL_TRANSFORMER_HPP

#include "../base_model/base_model.hpp"

class ProposedModelTransformerImpl : public BaseModel {
public:
    ProposedModelTransformerImpl() : ProposedModelTransformerImpl(SearchOptions()) {}
    explicit ProposedModelTransformerImpl(const SearchOptions& search_options);

    //インタンスから下のクラス変数を参照するための関数
    std::string modelPrefix() override { return "proposed_model_transformer"; }

private:
    //探索全体
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> search(std::vector<Position>& positions) override;

    //各部分の推論
    std::tuple<torch::Tensor, torch::Tensor> infer(const torch::Tensor& x, const std::vector<torch::Tensor>& history);
    torch::Tensor positionalEncoding(int64_t pos) const;

    //transformer
    torch::nn::Linear first_compressor_{ nullptr };
    torch::nn::Transformer transformer_{ nullptr };
    torch::nn::Linear policy_head_{ nullptr };
    torch::nn::Linear value_head_{ nullptr };

    torch::Tensor positional_encoding_;
};
TORCH_MODULE(ProposedModelTransformer);

#endif //MIACIS_PROPOSED_MODEL_TRANSFORMER_HPP