#!/usr/bin/env python3
import argparse
import torch
import torch.jit
import torch.nn as nn


class HandEncoder(nn.Module):
    def __init__(self, channel_num) -> None:
        super().__init__()
        self.linear_ = nn.Linear(7, channel_num)

    def forward(self, hand):
        hand = hand.to(torch.float)
        hand = self.linear_(hand)
        hand = hand.unsqueeze(1)
        hand = hand.expand([hand.shape[0], 81, hand.shape[2]])
        return hand


class InputConverter(nn.Module):
    def __init__(self, channel_num) -> None:
        super().__init__()
        self.embedding_board_ = nn.Embedding(88, channel_num)
        self.hand_encoder_for_turn_player_ = HandEncoder(channel_num)
        self.hand_encoder_for_opp_player_ = HandEncoder(channel_num)

    def forward(self, x):
        # xのshapeは[batch_size, 95]
        # 先頭81個が盤面、それ以降が持ち駒(95 = 81 + 7 + 7)
        board, hand_for_turn, hand_for_opp = torch.split(x, [81, 7, 7], 1)
        board = self.embedding_board_(board)

        hand_for_turn = self.hand_encoder_for_turn_player_(hand_for_turn)
        hand_for_opp = self.hand_encoder_for_opp_player_(hand_for_opp)

        # 和で結合
        x = board + hand_for_turn + hand_for_opp
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_channel_num, block_num, channel_num, policy_channel_num, board_size):
        super(TransformerModel, self).__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(
            channel_num,
            nhead=8,
            dim_feedforward=channel_num * 4,
            norm_first=True,
            activation="gelu",
            batch_first=True,
            dropout=0.0)
        self.encoder_ = torch.nn.TransformerEncoder(encoder_layer, block_num)
        self.board_size = board_size
        square_num = board_size ** 2
        self.policy_head_ = nn.Linear(channel_num, policy_channel_num)
        self.value_head_ = nn.Linear(channel_num, 51)
        seq_len = square_num + 1  # value_tokenの分を加える
        self.positional_encoding_ = torch.nn.Parameter(torch.zeros([1, seq_len, channel_num]), requires_grad=True)
        self.input_converter_ = InputConverter(channel_num)

    def forward(self, x):
        x = self.input_converter_(x)
        value_token = torch.zeros([x.shape[0], 1, x.shape[2]]).to(x.device)
        x = torch.cat([x, value_token], dim=1)
        x = x + self.positional_encoding_

        x = self.encoder_(x)

        # policy_headへの入力は先頭81マスに相当する部分
        policy_x = x[:, :81]

        # value_headへの入力はvalue_tokenに相当する部分
        value_x = x[:, -1]

        policy = self.policy_head_(policy_x)
        value = self.value_head_(value_x)
        return policy, value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-game", default="shogi", choices=["shogi", "othello", "go"])
    parser.add_argument("-value_type", default="cat", choices=["sca", "cat"])
    parser.add_argument("--block_num", type=int, default=6)
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

    input_data = torch.tensor([[66,  0, 65,  0,  0,  0, 33,  0, 34,
                                67, 70, 65,  0,  0,  0, 33, 39, 35,
                                68,  0, 65,  0,  0,  0, 33,  0, 36,
                                69,  0, 65,  0,  0,  0, 33,  0, 37, 
                                72,  0, 65,  0,  0,  0, 33,  0, 40,
                                69,  0, 65,  0,  0,  0, 33,  0, 37,
                                68,  0, 65,  0,  0,  0, 33,  0, 36,
                                67, 71, 65,  0,  0,  0, 33, 38, 35,
                                66,  0, 65,  0,  0,  0, 33,  0, 34,
                                0, 0, 0, 0, 0, 0, 0, # 手番側の持ち駒 
                                0, 0, 0, 0, 0, 0, 0  # 非手番側の持ち駒
                                ]], dtype=torch.int64)
    script_model = torch.jit.trace(model, input_data)
    script_model = torch.jit.script(model)
    model_path = f"./{args.game}_cat_transformer_bl{args.block_num}_ch{args.channel_num}.model"
    script_model.save(model_path)
    print(f"{model_path}にパラメータを保存")


if __name__ == "__main__":
    main()
