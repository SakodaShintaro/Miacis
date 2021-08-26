#!/usr/bin/env python3
import argparse
import torch
from torch import nn
from generate_cnn_model import PolicyHead, ValueHead


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim):
        super().__init__()
        self.token_norm = nn.LayerNorm(dim)
        self.token_forward = FeedForward(num_patch, token_dim)
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim),
        )

    def forward(self, x):
        s = x
        x = self.token_norm(x)
        x = x.permute([0, 2, 1])
        x = self.token_forward(x)
        x = x.permute([0, 2, 1])
        x = x + s

        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Module):
    def __init__(self, in_channels, dim, board_size, depth):
        super().__init__()

        token_dim = dim
        channel_dim = dim * 2

        self.num_patch = board_size * board_size
        self.first_conv = nn.Conv2d(in_channels, dim, 1, 1)

        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))
        self.layer_norm = nn.LayerNorm(dim)
        self.dim = dim
        self.board_size = board_size
        self.policy_head_ = PolicyHead(dim, policy_channel_num)
        self.value_head_ = ValueHead(dim, 51)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.first_conv.forward(x)
        x = x.permute([0, 2, 3, 1])
        x = x.view([-1, self.board_size * self.board_size, self.dim])
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = x.permute([0, 2, 1])
        x = x.contiguous()
        x = x.view([batch_size, self.dim, self.board_size, self.board_size])
        policy = self.policy_head_.forward(x)
        value = self.value_head_.forward(x)
        return policy, value

    @torch.jit.export
    def getRepresentations(self, x):
        x = self.first_conv(x)
        x = x.permute([0, 2, 3, 1])
        x = x.view([-1, self.board_size * self.board_size, self.dim])
        result = []
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
            result.append(x)
        return result


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

    input_data = torch.randn([128, input_channel_num, board_size, board_size])
    script_model = torch.jit.trace(model, input_data)
    # script_model = torch.jit.script(model)
    model_path = f"./{args.game}_{args.value_type}_bl{args.block_num}_ch{args.channel_num}.model"
    script_model.save(model_path)

    print(f"{model_path}にパラメータを保存")

    model = torch.jit.load(model_path)
    reps = model.getRepresentations(input_data)
    for i, r in enumerate(reps, 1):
        m = r.mean([0, 2])
        m = (m * m).mean()
        v = r.var([0, 2]).mean()
        print(f"{i}\t{m.item():.4f}\t{v.item():.4f}")
