#!/usr/bin/env python3
import argparse
import torch
import torch.jit
import torch.nn as nn
from timm.models.vision_transformer import _init_vit_weights, Block
import math
from functools import partial
import torch.nn as nn
from timm.models.helpers import named_apply
from timm.models.layers import PatchEmbed, trunc_normal_


# 参考) timmのVisionTransformer
# https://github.com/rwightman/pytorch-image-models/blob/5f81d4de234f579bdc988e8346da14b37a3af160/timm/models/vision_transformer.py
class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x


class TimmVit(nn.Module):
    def __init__(self, input_channel_num, block_num, channel_num, policy_channel_num, board_size):
        super(TimmVit, self).__init__()
        nhead = None
        if channel_num == 256:
            nhead = 8
        elif channel_num == 384:
            nhead = 6
        elif channel_num == 512:
            nhead = 16
        elif channel_num == 768:
            nhead = 12
        self.encoder_ = VisionTransformer(
            img_size=9,
            patch_size=1,
            in_chans=input_channel_num,
            num_classes=0,
            embed_dim=channel_num,
            depth=block_num,
            num_heads=nhead)
        self.policy_head_ = nn.Conv2d(channel_num, policy_channel_num, 1, bias=True, padding=0)
        self.value_head_ = nn.Conv2d(channel_num, 51, 1, bias=True, padding=0)
        self.board_size = board_size

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.encoder_.forward(x)

        # 先頭81マスに相当するところはpolicyへ、残りの1個はvalueへ
        policy_x = x[:, 0:81]
        value_x = x[:, 81:82]

        policy_x = policy_x.permute([0, 2, 1])
        policy_x = policy_x.view([batch_size, policy_x.shape[1], self.board_size, self.board_size])
        policy = self.policy_head_(policy_x)
        value_x = value_x.permute([0, 2, 1])
        value_x = value_x.view([batch_size, value_x.shape[1], 1, 1])
        value = self.value_head_(value_x)
        value = value.squeeze(3).squeeze(2)
        return policy, value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-value_type", default="cat", choices=["sca", "cat"])
    parser.add_argument("--block_num", type=int, default=12)
    parser.add_argument("--channel_num", type=int, default=384)
    args = parser.parse_args()

    input_channel_num = 42
    board_size = 9
    policy_channel_num = 27

    model = TimmVit(input_channel_num, args.block_num, args.channel_num, policy_channel_num, board_size)

    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"パラメータ数 : {params:,}")

    input_data = torch.randn([16, input_channel_num, board_size, board_size])
    script_model = torch.jit.trace(model, input_data)
    script_model = torch.jit.script(model)
    model_path = f"./{args.game}_cat_transformer_bl{args.block_num}_ch{args.channel_num}.model"
    script_model.save(model_path)
    print(f"{model_path}にパラメータを保存")


if __name__ == "__main__":
    main()
