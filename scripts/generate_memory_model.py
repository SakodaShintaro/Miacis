#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import argparse


class Conv2DwithLayerNorm(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size):
        super(Conv2DwithLayerNorm, self).__init__()
        self.conv_ = nn.Conv2d(input_ch, output_ch, kernel_size, bias=False, padding=kernel_size // 2)
        self.norm_ = nn.LayerNorm((output_ch, 9, 9))

    def forward(self, x):
        t = self.conv_.forward(x)
        t = self.norm_.forward(t)
        return t


class Branch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Branch, self).__init__()
        hidden_channels = in_channels // 2
        self.conv0_ = Conv2DwithLayerNorm(in_channels, hidden_channels, kernel_size=kernel_size)
        self.conv1_ = Conv2DwithLayerNorm(hidden_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        x = self.conv0_(x)
        x = torch.relu(x)
        x = self.conv1_(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, hidden_channels, memory_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        sum_channels = hidden_channels + memory_channels
        self.conv_update_gate_ = Branch(sum_channels, memory_channels, kernel_size)
        self.conv_forget_gate_ = Branch(sum_channels, memory_channels, kernel_size)
        self.conv_update_value_ = Branch(sum_channels, memory_channels, kernel_size)
        self.conv_forward_ = Branch(sum_channels, hidden_channels, kernel_size)

    def forward(self, x, memory):
        cat0 = torch.cat([x, memory], dim=1)

        update_gate = torch.sigmoid(self.conv_update_gate_(cat0))
        forget_gate = torch.sigmoid(self.conv_forget_gate_(cat0))
        update_value = torch.tanh(self.conv_update_value_(cat0))

        memory = update_gate * update_value + forget_gate * memory

        cat1 = torch.cat([x, memory], dim=1)
        next_x = torch.relu(x + self.conv_forward_(cat1))

        return next_x, memory


class ResidualLayer(nn.Module):
    def __init__(self, block_num, hidden_channel_num, memory_channel_num, kernel_size=3):
        super(ResidualLayer, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(block_num):
            self.blocks.add_module(f"block{i}", ResidualBlock(hidden_channel_num, memory_channel_num, kernel_size))

    def forward(self, x, memory):
        for b in self.blocks:
            x, memory = b.forward(x, memory)
        return x, memory


class Encoder(nn.Module):
    def __init__(self, input_channel_num, block_num, channel_num, kernel_size=3, reduction=8):
        super(Encoder, self).__init__()
        self.hidden_channel_num = channel_num // 2
        self.memory_channel_num = channel_num - self.hidden_channel_num
        self.first_conv_and_norm_ = Conv2DwithLayerNorm(input_channel_num, self.hidden_channel_num, 3)
        self.layer_ = ResidualLayer(2, self.hidden_channel_num, self.memory_channel_num, kernel_size)
        self.iter_num_ = block_num // 2
        self.memory = None

    def forward(self, x):
        x = self.first_conv_and_norm_.forward(x)
        x = F.relu(x)
        memory = torch.zeros((x.shape[0], self.memory_channel_num, x.shape[2], x.shape[3]),
                             ).to(x.device)
        for _ in range(self.iter_num_):
            x, memory = self.layer_.forward(x, memory)
        return torch.cat((x, memory), dim=1)


class PolicyHead(nn.Module):
    def __init__(self, channel_num, policy_channel_num):
        super(PolicyHead, self).__init__()
        self.policy_conv_ = nn.Conv2d(channel_num, policy_channel_num, 1, bias=True, padding=0)

    def forward(self, x):
        policy = self.policy_conv_.forward(x)
        return policy


class ValueHead(nn.Module):
    def __init__(self, channel_num, board_size, unit_num, hidden_size=256):
        super(ValueHead, self).__init__()
        self.value_conv_and_norm_ = Conv2DwithLayerNorm(channel_num, channel_num, 1)
        self.hidden_size = channel_num * board_size * board_size
        self.value_linear0_ = nn.Linear(self.hidden_size, hidden_size)
        self.value_linear1_ = nn.Linear(hidden_size, unit_num)

    def forward(self, x):
        value = self.value_conv_and_norm_.forward(x)
        value = F.relu(value)
        value = value.view([-1, self.hidden_size])
        value = self.value_linear0_.forward(value)
        value = F.relu(value)
        value = self.value_linear1_.forward(value)
        return value


class EncodeHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(EncodeHead, self).__init__()
        self.linear0 = nn.Linear(in_features, hidden_features)
        self.linear1 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        y = F.avg_pool2d(x, [x.shape[2], x.shape[3]])
        y = y.view([-1, x.shape[1]])
        y = y.flatten(1)
        y = self.linear0(y)
        y = F.relu(y)
        y = self.linear1(y)
        return y


class ScalarNetwork(nn.Module):
    def __init__(self, input_channel_num, block_num, channel_num, policy_channel_num, board_size):
        super(ScalarNetwork, self).__init__()
        self.encoder_ = Encoder(input_channel_num, block_num, channel_num)
        self.policy_head_ = PolicyHead(channel_num, policy_channel_num)
        self.value_head_ = ValueHead(channel_num, board_size, 1)

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
        self.value_head_ = ValueHead(channel_num, board_size, 51)
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