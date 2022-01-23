#!/usr/bin/env python3
import argparse
import torch
import torch.jit
import torch.nn as nn
from generate_cnn_model import PolicyHead, ValueHead
from timm.models.layers import DropPath, trunc_normal_


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Mlp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        hidden_features = in_features * 4
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(0.)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()

        kernel_size = 5
        self.token_mixer = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm = LayerNormChannel(dim)
        self.mlp = Mlp(in_features=dim)

        # The following two techniques are useful to train deep PoolFormers.
        drop_path = 0.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()

        self.use_layer_scale = False
        layer_scale_init_value = 1e-5
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(1).unsqueeze(2)
                * self.token_mixer(x))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(1).unsqueeze(2)
                * self.mlp(self.norm(x)))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(self.norm(x)))
        return x


class ConvNeXt(nn.Module):
    def __init__(self, input_channel_num, block_num, channel_num, policy_channel_num, board_size):
        super(ConvNeXt, self).__init__()
        self.first_conv1x1_ = nn.Conv2d(input_channel_num, channel_num, 1)
        self.blocks_ = [Block(channel_num) for _ in range(block_num)]
        self.blocks_ = nn.Sequential(*self.blocks_)
        self.policy_head_ = PolicyHead(channel_num, policy_channel_num)
        self.value_head_ = ValueHead(channel_num, 51)

    def encode(self, x):
        x = self.first_conv1x1_(x)
        x = self.blocks_(x)
        return x

    def decode(self, representation):
        policy = self.policy_head_(representation)
        value = self.value_head_(representation)
        return policy, value

    def forward(self, x):
        return self.decode(self.encode(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-game", default="shogi", choices=["shogi", "othello", "go"])
    parser.add_argument("-value_type", default="cat", choices=["sca", "cat"])
    parser.add_argument("--block_num", type=int, default=10)
    parser.add_argument("--channel_num", type=int, default=256)
    args = parser.parse_args()

    if args.game == "shogi":
        input_channel_num = 42
        board_size = 9
        policy_channel_num = 27
    elif args.game == "othello":
        input_channel_num = 2
        board_size = 8
        policy_channel_num = 2
    else:
        exit(1)

    model = ConvNeXt(input_channel_num, args.block_num, args.channel_num, policy_channel_num, board_size)

    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"パラメータ数 : {params:,}")

    input_data = torch.randn([8, input_channel_num, board_size, board_size])
    script_model = torch.jit.trace(model, input_data)
    script_model = torch.jit.script(model)
    model_path = f"./{args.game}_cat_convnext_bl{args.block_num}_ch{args.channel_num}.model"
    script_model.save(model_path)
    print(f"{model_path}にパラメータを保存")


if __name__ == "__main__":
    main()
