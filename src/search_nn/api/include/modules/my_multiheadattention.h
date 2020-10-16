#ifndef MIACIS_MY_MULTIHEADATTENTION_H
#define MIACIS_MY_MULTIHEADATTENTION_H

#include <torch/torch.h>

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiheadAttention ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the MultiheadAttention function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.MultiheadAttention
/// to learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::MultiheadAttentionOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// MultiheadAttention model(MultiheadAttentionOptions(20, 10).bias(false));
/// ```
class MyMultiheadAttentionImpl : public torch::nn::Cloneable<MyMultiheadAttentionImpl> {
public:
    MyMultiheadAttentionImpl(int64_t embed_dim, int64_t num_heads)
        : MyMultiheadAttentionImpl(MultiheadAttentionOptions(embed_dim, num_heads)) {}
    explicit MyMultiheadAttentionImpl(const MultiheadAttentionOptions& options_);

    std::tuple<Tensor, Tensor> forward(const Tensor& query, const Tensor& key, const Tensor& value,
                                       const Tensor& key_padding_mask = {}, bool need_weights = true,
                                       const Tensor& attn_mask = {});

protected:
    FORWARD_HAS_DEFAULT_ARGS({ 3, AnyValue(Tensor()) }, { 4, AnyValue(true) }, { 5, AnyValue(Tensor()) })

public:
    void reset() override;

    void _reset_parameters();

    /// The options with which this `Module` was constructed.
    MultiheadAttentionOptions options;

    bool _qkv_same_embed_dim;
    Tensor in_proj_weight;
    Tensor in_proj_bias;
    Tensor bias_k;
    Tensor bias_v;
    Linear out_proj = nullptr;
    Tensor q_proj_weight;
    Tensor k_proj_weight;
    Tensor v_proj_weight;
    int64_t head_dim;
};

/// A `ModuleHolder` subclass for `MultiheadAttentionImpl`.
/// See the documentation for `MultiheadAttentionImpl` class to learn what methods it
/// provides, and examples of how to use `MultiheadAttention` with `torch::nn::MultiheadAttentionOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(MyMultiheadAttention);

} // namespace nn
} // namespace torch

#endif //MIACIS_MY_MULTIHEADATTENTION_H