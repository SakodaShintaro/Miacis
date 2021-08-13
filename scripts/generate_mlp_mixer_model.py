#!/usr/bin/env python3
import argparse
import torch
from torch import nn
from generate_cnn_model import PolicyHead, ValueHead


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()
        self.token_norm = nn.LayerNorm(dim)
        self.token_forward = FeedForward(num_patch, token_dim, dropout)
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        s = x
        x = self.token_norm(x)
        x = x.permute([0, 2, 1])
        x = self.token_forward(x)
        x = x.permute([0, 2, 1])
        x += s

        x = x + self.channel_mix(x)
        return x


class Conv2DwithBatchNorm(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size):
        super(Conv2DwithBatchNorm, self).__init__()
        self.conv_ = nn.Conv2d(input_ch, output_ch, kernel_size, bias=False, padding=kernel_size // 2)
        self.norm_ = nn.BatchNorm2d(output_ch)

    def forward(self, x):
        t = self.conv_.forward(x)
        t = self.norm_.forward(t)
        return t


class MLPMixer(nn.Module):
    def __init__(self, in_channels, dim, board_size, depth, token_dim, channel_dim):
        super().__init__()

        self.num_patch = board_size ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, 1, 1),
        )

        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))
        self.layer_norm = nn.LayerNorm(dim)
        self.dim = dim
        self.board_size = board_size
        self.policy_head_ = PolicyHead(dim, board_size * board_size * policy_channel_num)
        self.value_head_ = ValueHead(dim, 51)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.to_patch_embedding(x)
        x = x.permute([0, 2, 3, 1])
        x = x.view([-1, self.board_size ** 2, self.dim])
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        print(x.shape)
        x = x.permute([0, 2, 1])
        print(x.shape)
        # x = x.contiguous()
        x = x.view([batch_size, self.dim, self.board_size, self.board_size])
        print(x.shape)
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

    model = MLPMixer(in_channels=input_channel_num, board_size=board_size, dim=args.channel_num, depth=args.block_num,
                     token_dim=256, channel_dim=512)

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
