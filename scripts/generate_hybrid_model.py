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
    def __init__(self, channel_num: int) -> None:
        super(ResidualBlock, self).__init__()
        kernel_size = 3
        reduction = 8
        self.conv_and_norm0_ = Conv2DwithBatchNorm(channel_num, channel_num, kernel_size)
        self.conv_and_norm1_ = Conv2DwithBatchNorm(channel_num, channel_num, kernel_size)
        self.linear0_ = nn.Linear(channel_num, channel_num // reduction, bias=False)
        self.linear1_ = nn.Linear(channel_num // reduction, channel_num, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x
        t = self.conv_and_norm0_.forward(t)
        t = F.relu(t)
        t = self.conv_and_norm1_.forward(t)

        y = F.adaptive_avg_pool2d(t, [1, 1])
        y = y.view([-1, t.shape[1]])
        y = self.linear0_.forward(y)
        y = F.relu(y)
        y = self.linear1_.forward(y)
        y = torch.sigmoid(y)
        y = y.view([-1, t.shape[1], 1, 1])
        t = t * y

        t = F.relu(x + t)
        return t


class TransformerBlock(nn.Module):
    def __init__(self, channel_num: int) -> None:
        super().__init__()
        self.layer_ = torch.nn.TransformerEncoderLayer(
            channel_num,
            nhead=8,
            dim_feedforward=channel_num * 4,
            norm_first=True,
            activation="gelu",
            dropout=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view([b, c, h * w])
        x = x.permute([2, 0, 1])
        x = self.layer_(x)
        x = x.permute([1, 2, 0])
        x = x.view([b, c, h, w])
        return x


class MixedBlock(nn.Module):
    # ResBlock2個とTransformerBlock1個を繋げる
    def __init__(self, channel_num: int) -> None:
        super().__init__()
        self.residual_block1_ = ResidualBlock(channel_num)
        self.residual_block2_ = ResidualBlock(channel_num)
        self.transformer_block_ = TransformerBlock(channel_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual_block1_(x)
        x = self.residual_block2_(x)
        x = self.transformer_block_(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_channel_num, block_num, channel_num):
        super(Encoder, self).__init__()
        self.first_conv_and_norm_ = Conv2DwithBatchNorm(input_channel_num, channel_num, 3)
        self.blocks = nn.ModuleList()
        for i in range(block_num):
            self.blocks.add_module(f"block{i}", MixedBlock(channel_num))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_conv_and_norm_.forward(x)
        x = F.relu(x)
        for b in self.blocks:
            x = b.forward(x)
        return x


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
        value = F.adaptive_avg_pool2d(value, [1, 1])
        value = value.view([-1, value.shape[1]])
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


class HybridNetwork(nn.Module):
    def __init__(self, input_channel_num, block_num, channel_num, policy_channel_num, board_size):
        super(HybridNetwork, self).__init__()
        self.encoder_ = Encoder(input_channel_num, block_num, channel_num)
        self.policy_head_ = PolicyHead(channel_num, policy_channel_num)
        self.value_head_ = ValueHead(channel_num, 51)

    def forward(self, x):
        x = self.encoder_.forward(x)
        policy = self.policy_head_.forward(x)
        value = self.value_head_.forward(x)
        return policy, value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-game", default="shogi", choices=["shogi", "othello", "go"])
    parser.add_argument("-value_type", default="cat", choices=["sca", "cat"])
    parser.add_argument("--block_num", type=int, default=5)
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
        model = HybridNetwork(input_channel_num, args.block_num, args.channel_num, policy_channel_num, board_size)

    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"パラメータ数 : {params:,}")

    input_data = torch.randn([16, input_channel_num, board_size, board_size])
    script_model = torch.jit.trace(model, input_data)
    script_model = torch.jit.script(model)
    model_path = f"./{args.game}_{args.value_type}_hybrid_bl{args.block_num}_ch{args.channel_num}.model"
    script_model.save(model_path)
    print(f"{model_path}にパラメータを保存")

    # model = torch.jit.load(model_path)
    # out = model(input_data)
    # print(out[0].shape, out[1].shape)


if __name__ == "__main__":
    main()
