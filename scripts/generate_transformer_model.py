#!/usr/bin/env python3
import argparse
import torch
import torch.jit
import torch.nn as nn
from timm.models.vision_transformer import Mlp
from generate_cnn_model import PolicyHead, ValueHead


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.head_dim = head_dim
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute([0, 2, 1, 3])
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute([0, 2, 1, 3])
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute([0, 2, 1, 3])

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim, elementwise_affine=False)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim, elementwise_affine=False)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_channel_num, block_num, channel_num, policy_channel_num, board_size):
        super(TransformerModel, self).__init__()
        self.first_encoding_ = torch.nn.Linear(input_channel_num, channel_num)
        nhead = None
        if channel_num == 256:
            nhead = 8
        elif channel_num == 384:
            nhead = 6
        elif channel_num == 512:
            nhead = 16
        elif channel_num == 768:
            nhead = 12
        elif channel_num == 1024:
            nhead = 16
        self.encoder_ = nn.Sequential(*[Block(dim=channel_num, num_heads=nhead) for _ in range(block_num)])
        self.channel_num = channel_num
        self.board_size = board_size
        square_num = board_size ** 2
        self.policy_head_ = PolicyHead(channel_num, policy_channel_num)
        self.value_head_ = ValueHead(channel_num, 51)
        self.positional_encoding_ = torch.nn.Parameter(torch.zeros([square_num, 1, channel_num]), requires_grad=True)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view([batch_size, x.shape[1], x.shape[2] * x.shape[3]])
        x = x.permute([2, 0, 1])
        x = self.first_encoding_(x)
        x = x + self.positional_encoding_

        x = self.encoder_(x)
        x = x.permute([1, 2, 0])
        x = x.view([batch_size, self.channel_num, self.board_size, self.board_size])

        policy = self.policy_head_.forward(x)
        value = self.value_head_.forward(x)
        policy = policy.flatten(1)
        return torch.cat([policy, value], dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-value_type", default="cat", choices=["sca", "cat"])
    parser.add_argument("--block_num", type=int, default=12)
    parser.add_argument("--channel_num", type=int, default=384)
    args = parser.parse_args()

    input_channel_num = 42
    board_size = 9
    policy_channel_num = 27

    model = TransformerModel(input_channel_num, args.block_num, args.channel_num, policy_channel_num, board_size)

    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"パラメータ数 : {params:,}")

    input_data = torch.randn([16, input_channel_num, board_size, board_size])
    script_model = torch.jit.trace(model, input_data)
    script_model = torch.jit.script(model)
    model_path = f"./shogi_cat_transformer_bl{args.block_num}_ch{args.channel_num}.ts"
    script_model.save(model_path)
    print(f"{model_path}にパラメータを保存")
    out = script_model(input_data)
    print(out.shape)


if __name__ == "__main__":
    main()
