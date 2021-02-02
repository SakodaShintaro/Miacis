#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-game", choices=["shogi", "othello"])
parser.add_argument("-value_type", choices=["sca", "cat"])
parser.add_argument("--block_num", type=int, default=10)
parser.add_argument("--channel_num", type=int, default=128)
args = parser.parse_args()

REDUCTION = 8
KERNEL_SIZE = 3

if args.game == "shogi":
    INPUT_CHANNEL_NUM = 42
    BOARD_SIZE = 9
    POLICY_CHANNEL_NUM = 27
    VALUE_HIDDEN_NUM = 256
elif args.game == "othello":
    INPUT_CHANNEL_NUM = 2
    BOARD_SIZE = 8
    POLICY_CHANNEL_NUM = 2
    VALUE_HIDDEN_NUM = 256


class Conv2DwithBatchNorm(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size):
        super(Conv2DwithBatchNorm, self).__init__()
        self.conv_ = nn.Conv2d(input_ch, output_ch, kernel_size, padding=kernel_size // 2)
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


class ScalarNetwork(nn.Module):
    def __init__(self, channel_num):
        super(ScalarNetwork, self).__init__()

        self.first_conv_and_norm_ = Conv2DwithBatchNorm(INPUT_CHANNEL_NUM, channel_num, 3)
        self.blocks = nn.Sequential()
        for i in range(args.block_num):
            self.blocks.add_module(f"block{i}", ResidualBlock(channel_num, KERNEL_SIZE, REDUCTION))
        self.policy_conv_ = nn.Conv2d(channel_num, POLICY_CHANNEL_NUM, 1, bias=True, padding=0)
        self.value_conv_and_norm_ = Conv2DwithBatchNorm(channel_num, channel_num, 1)
        self.value_linear0_ = nn.Linear(BOARD_SIZE * BOARD_SIZE * channel_num, VALUE_HIDDEN_NUM)
        self.value_linear1_ = nn.Linear(VALUE_HIDDEN_NUM, 1)

    def forward(self, x):
        x = self.first_conv_and_norm_.forward(x)
        x = F.relu(x)

        x = self.blocks.forward(x)
        policy = self.policy_conv_.forward(x)

        value = self.value_conv_and_norm_.forward(x)
        value = F.relu(value)
        value = value.view([-1, args.channel_num * BOARD_SIZE * BOARD_SIZE])
        value = self.value_linear0_.forward(value)
        value = F.relu(value)
        value = self.value_linear1_.forward(value)
        value = torch.tanh(value)

        return policy, value

class CategoricalNetwork(nn.Module):
    def __init__(self, channel_num):
        super(CategoricalNetwork, self).__init__()

        self.first_conv_and_norm_ = Conv2DwithBatchNorm(INPUT_CHANNEL_NUM, channel_num, 3)
        self.blocks = nn.Sequential()
        for i in range(args.block_num):
            self.blocks.add_module(f"block{i}", ResidualBlock(channel_num, KERNEL_SIZE, REDUCTION))
        self.policy_conv_ = nn.Conv2d(channel_num, POLICY_CHANNEL_NUM, 1, bias=True, padding=0)
        self.value_conv_and_norm_ = Conv2DwithBatchNorm(channel_num, channel_num, 1)
        self.value_linear0_ = nn.Linear(BOARD_SIZE * BOARD_SIZE * channel_num, VALUE_HIDDEN_NUM)
        self.value_linear1_ = nn.Linear(VALUE_HIDDEN_NUM, 51)

    def forward(self, x):
        x = self.first_conv_and_norm_.forward(x)
        x = F.relu(x)

        x = self.blocks.forward(x)
        policy = self.policy_conv_.forward(x)

        value = self.value_conv_and_norm_.forward(x)
        value = F.relu(value)
        value = value.view([-1, args.channel_num * BOARD_SIZE * BOARD_SIZE])
        value = self.value_linear0_.forward(value)
        value = F.relu(value)
        value = self.value_linear1_.forward(value)

        return policy, value

model = None
if args.value_type == "sca":
    model = ScalarNetwork(args.channel_num)
elif args.value_type == "cat":
    model = CategoricalNetwork(args.channel_num)
input_data = torch.randn([8, INPUT_CHANNEL_NUM, BOARD_SIZE, BOARD_SIZE])
script_model = torch.jit.trace(model, input_data)
# script_model = torch.jit.script(model)
script_model.save(f"./{args.game}_{args.value_type}_bl{args.block_num}_ch{args.channel_num}.model")
