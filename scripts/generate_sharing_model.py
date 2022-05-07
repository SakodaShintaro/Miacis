#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import argparse
from generate_cnn_model import PolicyHead, ValueHead, EncodeHead


class Conv2DwithLayerNorm(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size):
        super(Conv2DwithLayerNorm, self).__init__()
        self.conv_ = nn.Conv2d(input_ch, output_ch, kernel_size, bias=False, padding=kernel_size // 2)
        self.norm_ = nn.LayerNorm((output_ch, 9, 9))

    def forward(self, x):
        t = self.conv_.forward(x)
        t = self.norm_.forward(t)
        return t


class ResidualBlock(nn.Module):
    def __init__(self, channel_num, kernel_size, reduction):
        super(ResidualBlock, self).__init__()
        self.conv_and_norm0_ = Conv2DwithLayerNorm(channel_num, channel_num, kernel_size)
        self.conv_and_norm1_ = Conv2DwithLayerNorm(channel_num, channel_num, kernel_size)
        self.linear0_ = nn.Linear(channel_num, channel_num // reduction, bias=False)
        self.linear1_ = nn.Linear(channel_num // reduction, channel_num, bias=False)

    def forward(self, x):
        t = x
        t = self.conv_and_norm0_.forward(t)
        t = F.relu(t)
        t = self.conv_and_norm1_.forward(t)

        y = F.avg_pool2d(t, [t.shape[2], t.shape[3]])
        y = y.view([-1, t.shape[1]])
        y = self.linear0_.forward(y)
        y = F.relu(y)
        y = self.linear1_.forward(y)
        y = torch.sigmoid(y)
        y = y.view([-1, t.shape[1], 1, 1])
        t = t * y

        t = F.relu(x + t)
        return t


class ResidualLayer(nn.Module):
    def __init__(self, block_num, channel_num, kernel_size=3, reduction=8):
        super(ResidualLayer, self).__init__()
        self.blocks = nn.Sequential()
        for i in range(block_num):
            self.blocks.add_module(f"block{i}", ResidualBlock(channel_num, kernel_size, reduction))

    def forward(self, x):
        return self.blocks.forward(x)


class Encoder(nn.Module):
    def __init__(self, input_channel_num, block_num, channel_num, kernel_size=3, reduction=8):
        super(Encoder, self).__init__()
        self.first_conv_and_norm_ = Conv2DwithLayerNorm(input_channel_num, channel_num, 3)
        self.layer_ = ResidualLayer(2, channel_num, kernel_size, reduction)
        self.iter_num_ = block_num // 2

    def forward(self, x):
        x = self.first_conv_and_norm_.forward(x)
        x = F.relu(x)
        for _ in range(self.iter_num_):
            x = self.layer_.forward(x)
        return x


class ScalarNetwork(nn.Module):
    def __init__(self, input_channel_num, block_num, channel_num, policy_channel_num, board_size):
        super(ScalarNetwork, self).__init__()
        self.encoder_ = Encoder(input_channel_num, block_num, channel_num)
        self.policy_head_ = PolicyHead(channel_num, policy_channel_num)
        self.value_head_ = ValueHead(channel_num, 1)

    def forward(self, x):
        x = self.encoder_.forward(x)
        policy = self.policy_head_.forward(x)
        value = self.value_head_.forward(x)
        value = torch.tanh(value)
        return policy, value


class CategoricalNetwork(nn.Module):
    def __init__(self, input_channel_num, block_num, channel_num, policy_channel_num, board_size):
        super(CategoricalNetwork, self).__init__()
        self.encoder_ = Encoder(input_channel_num, block_num, channel_num)
        self.policy_head_ = PolicyHead(channel_num, policy_channel_num)
        self.value_head_ = ValueHead(channel_num, 51)
        self.encoder_head = EncodeHead(channel_num, channel_num, channel_num)

    def forward(self, x):
        x = self.encoder_.forward(x)
        policy = self.policy_head_.forward(x)
        value = self.value_head_.forward(x)
        return policy, value

    @torch.jit.export
    def encode(self, x):
        x = self.encoder_.forward(x)
        x = self.encoder_head.forward(x)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-value_type", default="cat", choices=["sca", "cat"])
    parser.add_argument("--block_num", type=int, default=10)
    parser.add_argument("--channel_num", type=int, default=256)
    args = parser.parse_args()

    input_channel_num = 42
    board_size = 9
    policy_channel_num = 27

    model = None
    if args.value_type == "sca":
        model = ScalarNetwork(input_channel_num, args.block_num, args.channel_num, policy_channel_num, board_size)
    elif args.value_type == "cat":
        model = CategoricalNetwork(input_channel_num, args.block_num, args.channel_num, policy_channel_num, board_size)

    # パラメータ数のカウント
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"パラメータ数 : {params:,}")
    input_data = torch.randn([8, input_channel_num, board_size, board_size])
    script_model = torch.jit.trace(model, input_data)
    script_model = torch.jit.script(model)
    model_path = f"./{args.game}_{args.value_type}_bl{args.block_num}_ch{args.channel_num}.model"
    script_model.save(model_path)
    print(f"{model_path}にパラメータを保存")


if __name__ == "__main__":
    main()
