#!/usr/bin/env python3
import argparse
import torch
from torch import nn
from generate_cnn_model import PolicyHead, ValueHead


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = dim * 2
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch):
        super().__init__()
        self.token_mix = FeedForward(num_patch)
        self.channel_mix = FeedForward(dim)

    def forward(self, x):
        # token mix
        s = x
        x = x.permute([0, 2, 1])
        x = self.token_mix(x)
        x = x.permute([0, 2, 1])
        x = x + s

        # channel mix
        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Module):
    def __init__(self, input_channel_num, block_num, channel_num, policy_channel_num, board_size):
        super().__init__()

        self.board_size = board_size
        self.num_patch = board_size * board_size
        self.first_conv = nn.Conv2d(input_channel_num, channel_num, 1, 1)

        self.mixer_blocks = nn.ModuleList([])
        for _ in range(block_num):
            self.mixer_blocks.append(MixerBlock(channel_num, self.num_patch))
        self.layer_norm = nn.LayerNorm(channel_num)
        self.dim = channel_num
        self.policy_head_ = PolicyHead(channel_num, policy_channel_num)
        self.value_head_ = ValueHead(channel_num, 51)

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
    parser.add_argument("-value_type", default="cat", choices=["sca", "cat"])
    parser.add_argument("--block_num", type=int, default=10)
    parser.add_argument("--channel_num", type=int, default=256)
    args = parser.parse_args()

    input_channel_num = 42
    board_size = 9
    policy_channel_num = 27

    model = MLPMixer(
        input_channel_num=input_channel_num,
        board_size=board_size,
        channel_num=args.channel_num,
        block_num=args.block_num,
        policy_channel_num=policy_channel_num)

    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"パラメータ数 : {params:,}")

    input_data = torch.randn([128, input_channel_num, board_size, board_size])
    script_model = torch.jit.trace(model, input_data)
    # script_model = torch.jit.script(model)
    model_path = f"./{args.game}_{args.value_type}_mlp_mixer_bl{args.block_num}_ch{args.channel_num}.model"
    script_model.save(model_path)

    print(f"{model_path}にパラメータを保存")
