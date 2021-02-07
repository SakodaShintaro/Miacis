#!/usr/bin/env python3

# !/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import argparse
import glob
import os
from natsort import natsorted

parser = argparse.ArgumentParser()
parser.add_argument("-source_dir", type=str, required=True)
parser.add_argument("-game", default="shogi", choices=["shogi", "othello"])
parser.add_argument("-value_type", default="cat", choices=["sca", "cat"])
parser.add_argument("--block_num", type=int, default=10)
parser.add_argument("--channel_num", type=int, default=128)
args = parser.parse_args()

REDUCTION = 8
KERNEL_SIZE = 3
VALUE_HIDDEN_NUM = 256

if args.game == "shogi":
    INPUT_CHANNEL_NUM = 42
    BOARD_SIZE = 9
    POLICY_CHANNEL_NUM = 27
elif args.game == "othello":
    INPUT_CHANNEL_NUM = 2
    BOARD_SIZE = 8
    POLICY_CHANNEL_NUM = 2


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


class CategoricalNetwork(nn.Module):
    def __init__(self, channel_num):
        super(CategoricalNetwork, self).__init__()
        self.state_first_conv_and_norm_ = Conv2DwithBatchNorm(INPUT_CHANNEL_NUM, channel_num, 3)
        self.blocks = nn.Sequential()
        for i in range(args.block_num):
            self.blocks.add_module(f"block{i}", ResidualBlock(channel_num, KERNEL_SIZE, REDUCTION))
        self.policy_conv_ = nn.Conv2d(channel_num, POLICY_CHANNEL_NUM, 1, bias=True, padding=0)

        self.value_conv_and_norm_ = Conv2DwithBatchNorm(channel_num, channel_num, 1)
        self.value_linear0_ = nn.Linear(BOARD_SIZE * BOARD_SIZE * channel_num, VALUE_HIDDEN_NUM)
        self.value_linear1_ = nn.Linear(VALUE_HIDDEN_NUM, 51)

    def forward(self, x):
        x = self.state_first_conv_and_norm_.forward(x)
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


def load_conv_and_norm(dst, src):
    dst.conv_.weight.data = src.conv_.weight.data
    dst.norm_.weight.data = src.norm_.weight.data
    dst.norm_.bias.data = src.norm_.bias.data
    dst.norm_.running_mean = src.norm_.running_mean
    dst.norm_.running_var = src.norm_.running_var


# インスタンス生成
model = CategoricalNetwork(args.channel_num)

# ディレクトリにある以下のprefixを持ったパラメータを用いて対局を行う
model_names = natsorted(glob.glob(f"{args.source_dir}/*.model"))

for source_model_name in model_names:
    source = torch.jit.load(source_model_name).cpu()

    # state_first
    load_conv_and_norm(model.state_first_conv_and_norm_, source.state_first_conv_and_norm_)

    # block
    for i, v in enumerate(model.__dict__["_modules"]["blocks"]):
        source_m = source.__dict__["_modules"][f"state_blocks_{i}"]
        load_conv_and_norm(v.conv_and_norm0_, source_m.conv_and_norm0_)
        load_conv_and_norm(v.conv_and_norm1_, source_m.conv_and_norm1_)
        v.linear0_.weight.data = source_m.linear0_.weight.data
        v.linear1_.weight.data = source_m.linear1_.weight.data

    # policy_conv
    model.policy_conv_.weight.data = source.policy_conv_.weight.data
    model.policy_conv_.bias.data = source.policy_conv_.bias.data

    # value_conv_norm_
    load_conv_and_norm(model.value_conv_and_norm_, source.value_conv_and_norm_)

    # value_linear
    model.value_linear0_.weight.data = source.value_linear0_.weight.data
    model.value_linear0_.bias.data = source.value_linear0_.bias.data
    model.value_linear1_.weight.data = source.value_linear1_.weight.data
    model.value_linear1_.bias.data = source.value_linear1_.bias.data

    input_data = torch.ones([1, INPUT_CHANNEL_NUM, BOARD_SIZE, BOARD_SIZE])
    model.eval()
    script_model = torch.jit.trace(model, input_data)
    # script_model = torch.jit.script(model)
    model_path = f"{args.game}_{os.path.basename(source_model_name)}"
    script_model.save(model_path)
    print(f"{model_path}にパラメータを保存")
