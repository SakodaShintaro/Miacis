#ifndef MIACIS_SIMPLE_MLP_HPP
#define MIACIS_SIMPLE_MLP_HPP

#include "../base_model/base_model.hpp"

class SimpleMLPImpl : public BaseModel {
public:
    SimpleMLPImpl() : SimpleMLPImpl(SearchOptions()) {}
    explicit SimpleMLPImpl(const SearchOptions& search_options);

    //インタンスから下のクラス変数を参照するための関数
    std::string modelPrefix() override { return "simple_mlp"; }

    //encoderとsim_policyをそれぞれ保存する関数
    void saveParts();

private:
    //探索
    std::vector<torch::Tensor> search(std::vector<Position>& positions) override;
};
TORCH_MODULE(SimpleMLP);

#endif //MIACIS_SIMPLE_MLP_HPP