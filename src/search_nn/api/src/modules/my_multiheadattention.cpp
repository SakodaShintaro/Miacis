#include "../../include/modules/my_multiheadattention.h"

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

MyMultiheadAttentionImpl::MyMultiheadAttentionImpl(const MultiheadAttentionOptions& options_)
    : Module("torch::nn::MyMultiheadAttention"), options(options_) {
    reset();
}

std::tuple<Tensor, Tensor> MyMultiheadAttentionImpl::forward(const Tensor& query, const Tensor& key, const Tensor& value,
                                                             const Tensor& key_padding_mask, bool need_weights,
                                                             const Tensor& attn_mask) {
    if (!_qkv_same_embed_dim) {
        return F::multi_head_attention_forward(query, key, value,
                                               F::MultiheadAttentionForwardFuncOptions(
                                                   /*embed_dim_to_check=*/options.embed_dim(),
                                                   /*num_heads=*/options.num_heads(),
                                                   /*in_proj_weight=*/in_proj_weight,
                                                   /*in_proj_bias=*/in_proj_bias,
                                                   /*bias_k=*/bias_k,
                                                   /*bias_v=*/bias_v,
                                                   /*add_zero_attn=*/options.add_zero_attn(),
                                                   /*dropout_p=*/options.dropout(),
                                                   /*out_proj_weight=*/out_proj->weight,
                                                   /*out_proj_bias=*/out_proj->bias)
                                                   .training(is_training())
                                                   .key_padding_mask(key_padding_mask)
                                                   .need_weights(need_weights)
                                                   .attn_mask(attn_mask)
                                                   .use_separate_proj_weight(true)
                                                   .q_proj_weight(q_proj_weight)
                                                   .k_proj_weight(k_proj_weight)
                                                   .v_proj_weight(v_proj_weight));
    } else {
        return F::multi_head_attention_forward(query, key, value,
                                               F::MultiheadAttentionForwardFuncOptions(
                                                   /*embed_dim_to_check=*/options.embed_dim(),
                                                   /*num_heads=*/options.num_heads(),
                                                   /*in_proj_weight=*/in_proj_weight,
                                                   /*in_proj_bias=*/in_proj_bias,
                                                   /*bias_k=*/bias_k,
                                                   /*bias_v=*/bias_v,
                                                   /*add_zero_attn=*/options.add_zero_attn(),
                                                   /*dropout_p=*/options.dropout(),
                                                   /*out_proj_weight=*/out_proj->weight,
                                                   /*out_proj_bias=*/out_proj->bias)
                                                   .training(is_training())
                                                   .key_padding_mask(key_padding_mask)
                                                   .need_weights(need_weights)
                                                   .attn_mask(attn_mask));
    }
}

void MyMultiheadAttentionImpl::reset() {
    _qkv_same_embed_dim = options.kdim() == options.embed_dim() && options.vdim() == options.embed_dim();
    head_dim = options.embed_dim() / options.num_heads();
    TORCH_CHECK(head_dim * options.num_heads() == options.embed_dim(), "embed_dim must be divisible by num_heads");
    if (!_qkv_same_embed_dim) {
        q_proj_weight = register_parameter("q_proj_weight", torch::empty({ options.embed_dim(), options.embed_dim() }));
        k_proj_weight = register_parameter("k_proj_weight", torch::empty({ options.embed_dim(), options.kdim() }));
        v_proj_weight = register_parameter("v_proj_weight", torch::empty({ options.embed_dim(), options.vdim() }));
        register_parameter("in_proj_weight", {}, /*requires_grad=*/false);
    } else {
        in_proj_weight = register_parameter("in_proj_weight", torch::empty({ 3 * options.embed_dim(), options.embed_dim() }));
        register_parameter("q_proj_weight", {}, /*requires_grad=*/false);
        register_parameter("k_proj_weight", {}, /*requires_grad=*/false);
        register_parameter("v_proj_weight", {}, /*requires_grad=*/false);
    }
    if (options.bias()) {
        in_proj_bias = register_parameter("in_proj_bias", torch::empty(3 * options.embed_dim()));
    } else {
        register_parameter("in_proj_bias", {}, /*requires_grad=*/false);
    }
    out_proj = register_module("out_proj", Linear(LinearOptions(options.embed_dim(), options.embed_dim()).bias(options.bias())));
    if (options.add_bias_kv()) {
        bias_k = register_parameter("bias_k", torch::empty({ 1, 1, options.embed_dim() }));
        bias_v = register_parameter("bias_v", torch::empty({ 1, 1, options.embed_dim() }));
    } else {
        bias_k = {};
        bias_v = {};
    }
    _reset_parameters();
}

void MyMultiheadAttentionImpl::_reset_parameters() {
    using namespace torch::nn::init;
    if (_qkv_same_embed_dim) {
        xavier_uniform_(in_proj_weight);
    } else {
        xavier_uniform_(q_proj_weight);
        xavier_uniform_(k_proj_weight);
        xavier_uniform_(v_proj_weight);
    }
    if (in_proj_bias.defined()) {
        constant_(in_proj_bias, 0.);
        constant_(out_proj->bias, 0.);
    }
    if (bias_k.defined()) {
        xavier_normal_(bias_k);
    }
    if (bias_v.defined()) {
        xavier_normal_(bias_v);
    }
}

} // namespace nn
} // namespace torch
