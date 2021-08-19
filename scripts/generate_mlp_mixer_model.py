#!/usr/bin/env python3
import argparse
import torch
from torch import nn
from generate_cnn_model import PolicyHead, ValueHead


class MixerBlock(nn.Module):
    def __init__(self, dim, board_size, token_dim, channel_dim):
        super().__init__()
        self.act = nn.GELU()

        num_patch = board_size ** 2
        self.board_size = board_size

        self.token_norm = nn.LayerNorm([dim, num_patch])
        self.token_linear1 = nn.Linear(num_patch, token_dim)
        self.token_linear2 = nn.Linear(token_dim, num_patch)

        self.channel_norm = nn.LayerNorm([dim, num_patch])
        self.channel_cnn1 = nn.Conv2d(dim, channel_dim, kernel_size=1, padding=0)
        self.channel_cnn2 = nn.Conv2d(channel_dim, dim, kernel_size=1, padding=0)

    def forward(self, x):
        s = x
        x = self.token_norm(x)
        x = self.token_linear1(x)
        x = self.act(x)
        x = self.token_linear2(x)
        x = x + s

        s = x
        x = self.channel_norm(x)
        x = x.view([-1, x.shape[1], self.board_size, self.board_size])
        x = self.channel_cnn1(x)
        x = self.act(x)
        x = self.channel_cnn2(x)
        x = x.view([-1, x.shape[1], self.board_size ** 2])
        x = x + s
        return x


class MLPMixer(nn.Module):
    def __init__(self, in_channels, dim, board_size, depth):
        super().__init__()

        token_dim = dim
        channel_dim = dim * 2

        self.num_patch = board_size ** 2
        self.first_conv = nn.Conv2d(in_channels, dim, 1, 1)

        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, board_size, token_dim, channel_dim))
        self.layer_norm = nn.LayerNorm([dim, self.num_patch])
        self.dim = dim
        self.board_size = board_size
        self.policy_head_ = PolicyHead(dim, policy_channel_num)
        self.value_head_ = ValueHead(dim, 51)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.first_conv(x)
        x = x.view([-1, self.dim, self.board_size ** 2])
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = x.permute([0, 2, 1])
        x = x.contiguous()
        x = x.view([batch_size, self.dim, self.board_size, self.board_size])
        policy = self.policy_head_.forward(x)
        value = self.value_head_.forward(x)
        return policy, value


if __name__ == "__main__":
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

    model = MLPMixer(in_channels=input_channel_num, board_size=board_size, dim=args.channel_num, depth=args.block_num)

    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"パラメータ数 : {params:,}")

    model.cuda()
    input_data = torch.randn([512, input_channel_num, board_size, board_size]).cuda()
    script_model = torch.jit.trace(model, input_data)
    # script_model = torch.jit.script(model)
    model_path = f"./{args.game}_{args.value_type}_bl{args.block_num}_ch{args.channel_num}.model"
    script_model.save(model_path)

    print(f"{model_path}にパラメータを保存")
