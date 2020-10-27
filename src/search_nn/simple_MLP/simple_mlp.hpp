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
    void save();

private:
    //探索
    std::vector<torch::Tensor> search(std::vector<Position>& positions) override;

    //入力として局面の特徴量を並べたvectorを受け取ってPolicyとValueに対応するTensorを返す関数
    torch::Tensor forward(const torch::Tensor& x);

    //1局面について方策を推論する関数
    torch::Tensor inferPolicy(const Position& pos);
};
TORCH_MODULE(SimpleMLP);

#endif //MIACIS_SIMPLE_MLP_HPP