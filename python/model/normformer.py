#!/usr/bin/env python3
import argparse
import torch
import torch.jit
import torch.nn as nn
from .vit_channel_num_to_nhead import channel_num_to_nhead


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerBlock, self).__init__()
        dim_feedforward = d_model * 4
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0, batch_first=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        layer_norm_eps = 1e-5
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm4 = nn.LayerNorm(dim_feedforward, eps=layer_norm_eps)

        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        x = src
        x = x + self._sa_block(x)
        x = x + self._ff_block(self.norm3(x))
        return x

    # self-attention block
    def _sa_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        x = self.self_attn(x, x, x,
                           attn_mask=None,
                           key_padding_mask=None,
                           need_weights=False)[0]
        x = self.norm2(x)
        return x

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm3(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.norm4(x)
        x = self.linear2(x)
        return x


class NormFormer(nn.Module):
    def __init__(self, input_channel_num, block_num, channel_num, policy_channel_num, board_size):
        super(NormFormer, self).__init__()
        self.first_encoding_ = torch.nn.Linear(input_channel_num, channel_num)
        nhead = channel_num_to_nhead[channel_num]
        self.encoder_ = [TransformerBlock(channel_num, nhead) for _ in range(block_num)]
        self.encoder_ = nn.Sequential(*self.encoder_)
        self.board_size = board_size
        square_num = board_size ** 2
        self.policy_head_ = nn.Conv2d(channel_num, policy_channel_num, 1, bias=True, padding=0)
        self.value_head_ = nn.Conv2d(channel_num, 51, 1, bias=True, padding=0)
        self.positional_encoding_ = torch.nn.Parameter(torch.zeros([square_num, 1, channel_num]), requires_grad=True)
        self.value_token_ = torch.nn.Parameter(torch.zeros([1, 1, channel_num]), requires_grad=True)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view([batch_size, x.shape[1], x.shape[2] * x.shape[3]])
        x = x.permute([2, 0, 1])
        x = self.first_encoding_(x)
        x = x + self.positional_encoding_

        value_token = self.value_token_.expand((1, batch_size, x.shape[2]))
        x = torch.cat([x, value_token], dim=0)

        x = self.encoder_(x)

        # 先頭81マスに相当するところはpolicyへ、残りの1個はvalueへ
        policy_x = x[0:81]
        value_x = x[81:82]

        policy_x = policy_x.permute([1, 2, 0])
        policy_x = policy_x.view([batch_size, policy_x.shape[1], self.board_size, self.board_size])
        policy = self.policy_head_(policy_x)
        value_x = value_x.permute([1, 2, 0])
        value_x = value_x.view([batch_size, value_x.shape[1], 1, 1])
        value = self.value_head_(value_x)
        value = value.squeeze(3).squeeze(2)
        return policy, value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-value_type", default="cat", choices=["sca", "cat"])
    parser.add_argument("--block_num", type=int, default=12)
    parser.add_argument("--channel_num", type=int, default=384)
    args = parser.parse_args()

    input_channel_num = 42
    board_size = 9
    policy_channel_num = 27

    model = NormFormer(input_channel_num, args.block_num, args.channel_num, policy_channel_num, board_size)

    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"パラメータ数 : {params:,}")

    input_data = torch.randn([16, input_channel_num, board_size, board_size])
    script_model = torch.jit.trace(model, input_data)
    script_model = torch.jit.script(model)
    model_path = f"./{args.game}_cat_transformer_bl{args.block_num}_ch{args.channel_num}.model"
    script_model.save(model_path)
    print(f"{model_path}にパラメータを保存")

    model = torch.jit.load(model_path)
    out = model(input_data)
    print(out[0].shape, out[1].shape)


if __name__ == "__main__":
    main()
