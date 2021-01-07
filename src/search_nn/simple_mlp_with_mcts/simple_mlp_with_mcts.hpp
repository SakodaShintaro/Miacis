#ifndef MIACIS_SIMPLE_MLP_WITH_MCTS_HPP
#define MIACIS_SIMPLE_MLP_WITH_MCTS_HPP

#include "../base_model/base_model.hpp"

class SimpleMLPWithMCTSImpl : public BaseModel {
public:
    SimpleMLPWithMCTSImpl() : SimpleMLPWithMCTSImpl(SearchOptions()) {}
    explicit SimpleMLPWithMCTSImpl(const SearchOptions& search_options);

    //インタンスから下のクラス変数を参照するための関数
    std::string modelPrefix() override { return "simple_mlp"; }

private:
    //探索
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> search(std::vector<Position>& positions) override;
};
TORCH_MODULE(SimpleMLPWithMCTS);

#endif //MIACIS_SIMPLE_MLP_WITH_MCTS_HPP
