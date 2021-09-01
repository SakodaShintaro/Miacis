#!/usr/bin/env python3
import argparse
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from generate_cnn_model import PolicyHead, ValueHead


class TransformerModel(nn.Module):
    def __init__(self, input_channel_num, layer_num, channel_num, policy_channel_num, board_size):
        super(TransformerModel, self).__init__()
        self.first_encoding_ = torch.nn.Linear(input_channel_num, channel_num)
        encoder_layer = torch.nn.TransformerEncoderLayer(channel_num, nhead=8, dim_feedforward=channel_num * 4)
        self.encoder_ = torch.nn.TransformerEncoder(encoder_layer, layer_num)
        self.board_size = board_size
        square_num = board_size ** 2
        self.policy_head_ = PolicyHead(channel_num, policy_channel_num)
        self.value_head_ = ValueHead(channel_num, 51)
        self.positional_encoding_ = torch.nn.Parameter(torch.zeros([square_num, 1, channel_num]), requires_grad=True)

    def encode(self, x):
        x = x.view([x.shape[0], x.shape[1], x.shape[2] * x.shape[3]])
        x = x.permute([2, 0, 1])
        x = self.first_encoding_(x)
        x = F.relu(x)
        x = x + self.positional_encoding_
        x = self.encoder_(x)
        x = x.permute([1, 2, 0])

        x = x.view([x.shape[0], x.shape[1], self.board_size, self.board_size])
        return x

    def decode(self, representation):
        policy = self.policy_head_(representation)
        value = self.value_head_(representation)
        return policy, value

    def forward(self, x):
        return self.decode(self.encode(x))


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

    model = TransformerModel(input_channel_num, args.block_num, args.channel_num, policy_channel_num, board_size)

    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"パラメータ数 : {params:,}")

    input_data = torch.randn([8, input_channel_num, board_size, board_size])
    script_model = torch.jit.trace(model, input_data)
    script_model = torch.jit.script(model)
    model_path = f"./{args.game}_cat_transformer_bl{args.block_num}_ch{args.channel_num}.model"
    script_model.save(model_path)
    print(f"{model_path}にパラメータを保存")


if __name__ == "__main__":
    main()
