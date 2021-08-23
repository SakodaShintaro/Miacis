#ifndef MIACIS_LIBTORCH_MODEL_HPP
#define MIACIS_LIBTORCH_MODEL_HPP

#include "model_common.hpp"

static constexpr int64_t SHARE_BLOCK_NUM = 2;
static constexpr int64_t DEFAULT_LOOP_NUM = BLOCK_NUM / SHARE_BLOCK_NUM;

//畳み込みとNormalizationをまとめたユニット
class Conv2DwithNormImpl : public torch::nn::Module {
public:
    Conv2DwithNormImpl(int64_t input_ch, int64_t output_ch, int64_t kernel_size);
    torch::Tensor forward(const torch::Tensor& x);

private:
    torch::nn::Conv2d conv_{ nullptr };
    torch::nn::LayerNorm norm_{ nullptr };
};
TORCH_MODULE(Conv2DwithNorm);

//残差ブロック:SENetの構造を利用
class ResidualBlockImpl : public torch::nn::Module {
public:
    ResidualBlockImpl(int64_t channel_num, int64_t kernel_size, int64_t reduction);
    torch::Tensor forward(const torch::Tensor& x);

private:
    Conv2DwithNorm conv_and_norm0_{ nullptr };
    Conv2DwithNorm conv_and_norm1_{ nullptr };
    torch::nn::Linear linear0_{ nullptr };
    torch::nn::Linear linear1_{ nullptr };
};
TORCH_MODULE(ResidualBlock);

//Valueヘッド
class ValueHeadImpl : public torch::nn::Module {
public:
    ValueHeadImpl(int64_t in_channels, int64_t out_dim, int64_t hidden_dim);
    torch::Tensor forward(const torch::Tensor& x);

private:
    Conv2DwithNorm conv_and_norm_{ nullptr };
    torch::nn::Linear linear0_{ nullptr };
    torch::nn::Linear linear1_{ nullptr };
};
TORCH_MODULE(ValueHead);

//使用する全体のニューラルネットワーク
class NetworkImpl : public torch::nn::Module {
public:
    NetworkImpl();
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(const torch::Tensor& x, int64_t loop_num = DEFAULT_LOOP_NUM);
    torch::Tensor firstEncode(const torch::Tensor& x);
    torch::Tensor applyOneLoop(const torch::Tensor& x);
    torch::Tensor encode(const torch::Tensor& x, int64_t loop_num = DEFAULT_LOOP_NUM);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> decode(const torch::Tensor& representation);
    std::vector<torch::Tensor> getRepresentations(const torch::Tensor& x, int64_t loop_num = DEFAULT_LOOP_NUM);

private:
    Conv2DwithNorm first_conv_and_norm_{ nullptr };
    std::vector<ResidualBlock> blocks_;
    torch::nn::Conv2d policy_head_{ nullptr };
    ValueHead value_head_{ nullptr };
    ValueHead ponder_head_{ nullptr };
};
TORCH_MODULE(Network);

class LibTorchModel {
public:
    LibTorchModel() : device_(torch::kCPU) {}
    void load(const std::string& model_path, int64_t gpu_id);
    void save(const std::string& model_path);
    std::array<torch::Tensor, LOSS_TYPE_NUM> loss(const std::vector<LearningData>& data);
    std::array<torch::Tensor, LOSS_TYPE_NUM> validLoss(const std::vector<LearningData>& data,
                                                       int64_t loop_num = DEFAULT_LOOP_NUM);
    std::vector<std::array<torch::Tensor, LOSS_TYPE_NUM>> validLosses(const std::vector<LearningData>& data,
                                                                      int64_t loop_num = DEFAULT_LOOP_NUM);

    //MixUpを行って損失を返す関数
    std::array<torch::Tensor, LOSS_TYPE_NUM> mixUpLoss(const std::vector<LearningData>& data, float alpha);

    torch::Tensor contrastiveLoss(const std::vector<LearningData>& data);

    std::vector<torch::Tensor> parameters();

    void train() { network_->train(); }
    void eval() { network_->eval(); }

    std::vector<torch::Tensor> getRepresentations(const std::vector<LearningData>& data);
    std::vector<torch::Tensor> getRepresentations(const torch::Tensor& x);

private:
    Network network_;
    torch::Device device_;
};

#endif //MIACIS_LIBTORCH_MODEL_HPP