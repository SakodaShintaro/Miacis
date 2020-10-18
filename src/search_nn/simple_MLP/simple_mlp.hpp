#ifndef MIACIS_SIMPLE_MLP_HPP
#define MIACIS_SIMPLE_MLP_HPP

#include "../base_model/base_model.hpp"

class SimpleMLPImpl : public BaseModel {
public:
    SimpleMLPImpl() : SimpleMLPImpl(SearchOptions()) {}
    explicit SimpleMLPImpl(SearchOptions search_options);

    //root局面について探索を行って一番良い指し手を返す関数
    Move think(Position& root, int64_t time_limit) override;

    //ミニバッチデータに対して損失を計算する関数(現在のところバッチサイズは1のみに対応)
    std::vector<torch::Tensor> loss(const std::vector<LearningData>& data) override;

    //インタンスから下のクラス変数を参照するための関数
    std::string modelPrefix() override { return "simple_mlp"; }

    //encoderとsim_policyをそれぞれ保存する関数
    void save();

private:
    //入力として局面の特徴量を並べたvectorを受け取ってPolicyとValueに対応するTensorを返す関数
    torch::Tensor forward(const torch::Tensor& x);

    //1局面について方策を推論する関数
    torch::Tensor inferPolicy(const Position& pos);
};
TORCH_MODULE(SimpleMLP);

#endif //MIACIS_SIMPLE_MLP_HPP