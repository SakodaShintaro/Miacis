#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import argparse


class Conv2DwithBatchNorm(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size):
        super(Conv2DwithBatchNorm, self).__init__()
        self.conv_ = nn.Conv2d(input_ch, output_ch, kernel_size, bias=False, padding=kernel_size // 2)
        self.norm_ = nn.BatchNorm2d(output_ch)

    def forward(self, x):
        t = self.conv_.forward(x)
        t = self.norm_.forward(t)
        return t


class ResidualBlock(nn.Module):
    def __init__(self, channel_num, kernel_size, reduction):
        super(ResidualBlock, self).__init__()
        self.conv_and_norm0_ = Conv2DwithBatchNorm(channel_num, channel_num, kernel_size)
        self.conv_and_norm1_ = Conv2DwithBatchNorm(channel_num, channel_num, kernel_size)
        self.linear0_ = nn.Linear(channel_num, channel_num // reduction, bias=False)
        self.linear1_ = nn.Linear(channel_num // reduction, channel_num, bias=False)

    def forward(self, x):
        t = x
        t = self.conv_and_norm0_.forward(t)
        t = F.relu(t)
        t = self.conv_and_norm1_.forward(t)

        y = F.adaptive_avg_pool2d(t, [1, 1])
        y = y.flatten(1)
        y = self.linear0_.forward(y)
        y = F.relu(y)
        y = self.linear1_.forward(y)
        y = torch.sigmoid(y)
        y = y.unsqueeze(2)
        y = y.unsqueeze(2)
        t = t * y

        t = F.relu(x + t)
        return t


class Encoder(nn.Module):
    def __init__(self, input_channel_num, block_num, channel_num, kernel_size=3, reduction=8):
        super(Encoder, self).__init__()
        self.first_conv_and_norm_ = Conv2DwithBatchNorm(input_channel_num, channel_num, 3)
        self.blocks = nn.ModuleList()
        for i in range(block_num):
            self.blocks.add_module(f"block{i}", ResidualBlock(channel_num, kernel_size, reduction))

    def forward(self, x):
        x = self.first_conv_and_norm_.forward(x)
        x = F.relu(x)
        for b in self.blocks:
            x = b.forward(x)
        return x

    def getRepresentations(self, x):
        x = self.first_conv_and_norm_.forward(x)
        x = F.relu(x)
        result = []
        for b in self.blocks:
            x = b.forward(x)
            result.append(x)
        return result


class PolicyHead(nn.Module):
    def __init__(self, channel_num, policy_channel_num):
        super(PolicyHead, self).__init__()
        self.policy_conv_ = nn.Conv2d(channel_num, policy_channel_num, 1, bias=True, padding=0)

    def forward(self, x):
        policy = self.policy_conv_.forward(x)
        return policy


class ValueHead(nn.Module):
    def __init__(self, channel_num, unit_num, hidden_size=256):
        super(ValueHead, self).__init__()
        self.value_conv_and_norm_ = Conv2DwithBatchNorm(channel_num, channel_num, 1)
        self.value_linear0_ = nn.Linear(channel_num, hidden_size)
        self.value_linear1_ = nn.Linear(hidden_size, unit_num)

    def forward(self, x):
        value = self.value_conv_and_norm_.forward(x)
        value = F.relu(value)
        # value = F.avg_pool2d(value, [int(value.shape[2]), int(value.shape[3])])
        value = F.adaptive_avg_pool2d(value, [1, 1])
        value = value.flatten(1)
        value = self.value_linear0_.forward(value)
        value = F.relu(value)
        value = self.value_linear1_.forward(value)
        return value


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

    def forward(self, x):
        x = self.encoder_.forward(x)
        policy = self.policy_head_.forward(x)
        value = self.value_head_.forward(x)
        policy = policy.flatten(1)
        return torch.cat([policy, value], dim=1)


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

    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"パラメータ数 : {params:,}")

    input_data = torch.randn([16, input_channel_num, board_size, board_size])
    script_model = torch.jit.trace(model, input_data)
    script_model = torch.jit.script(model)
    model_path = f"./shogi_{args.value_type}_bl{args.block_num}_ch{args.channel_num}.ts"
    script_model.save(model_path)
    print(f"{model_path}にパラメータを保存")
    out = script_model(input_data)
    print(out.shape)


if __name__ == "__main__":
    main()
