#!/usr/bin/env python3
import argparse
import torch
import torch.jit
import torch.nn as nn
from generate_cnn_model import PolicyHead


class HandEncoder(nn.Module):
    def __init__(self, channel_num) -> None:
        super().__init__()
        self.linear_pawn = nn.Linear(18, channel_num)
        self.linear_lance = nn.Linear(4, channel_num)
        self.linear_knight = nn.Linear(4, channel_num)
        self.linear_silver = nn.Linear(4, channel_num)
        self.linear_gold = nn.Linear(4, channel_num)
        self.linear_bishop = nn.Linear(2, channel_num)
        self.linear_rook = nn.Linear(2, channel_num)

    def forward(self, hand):
        hand = hand.to(torch.float)
        pawn, lance, knight, silver, gold, bishop, rook = torch.split(hand, [18, 4, 4, 4, 4, 2, 2], 1)
        pawn = self.linear_pawn(pawn).unsqueeze(1)
        lance = self.linear_lance(lance).unsqueeze(1)
        knight = self.linear_knight(knight).unsqueeze(1)
        silver = self.linear_silver(silver).unsqueeze(1)
        gold = self.linear_gold(gold).unsqueeze(1)
        bishop = self.linear_bishop(bishop).unsqueeze(1)
        rook = self.linear_rook(rook).unsqueeze(1)
        hand = torch.cat([pawn, lance, knight, silver, gold, bishop, rook], dim=1)
        return hand


class InputConverter(nn.Module):
    def __init__(self, channel_num) -> None:
        super().__init__()
        self.embedding_board_ = nn.Embedding(88, channel_num)
        self.hand_encoder_for_turn_player_ = HandEncoder(channel_num)
        self.hand_encoder_for_opp_player_ = HandEncoder(channel_num)

    def forward(self, x):
        # xのshapeは[batch_size, 157]
        # 先頭81個が盤面、それ以降が持ち駒(157 = 81 + 38 + 38)
        board, hand = torch.split(x, 81, 1)
        board = self.embedding_board_(board)

        hand_for_turn_player, hand_for_opp_player = torch.split(hand, 38, 1)
        hand_for_turn_player = self.hand_encoder_for_turn_player_(hand_for_turn_player)
        hand_for_opp_player = self.hand_encoder_for_opp_player_(hand_for_opp_player)
        x = torch.cat([board, hand_for_turn_player, hand_for_opp_player], dim=1)
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
            batch_first=True)
        self.encoder_ = torch.nn.TransformerEncoder(encoder_layer, block_num)
        self.board_size = board_size
        square_num = board_size ** 2
        self.policy_head_ = PolicyHead(channel_num, policy_channel_num)
        self.value_head_ = nn.Linear(channel_num, 51)
        seq_len = square_num + 7 * 2 + 1  # 持ち駒の分, value_tokenの分を加える
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
        policy_x = policy_x.permute([0, 2, 1]).contiguous()
        policy_x = policy_x.view([policy_x.shape[0], policy_x.shape[1], 9, 9])

        # value_headへの入力はvalue_tokenに相当する部分
        value_x = x[:, -1]

        policy = self.policy_head_(policy_x)
        value = self.value_head_(value_x)
        return policy, value


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

    input_data = torch.tensor([[66,  0, 65,  0,  0,  0, 33,  0, 34,
                                67, 70, 65,  0,  0,  0, 33, 39, 35,
                                68,  0, 65,  0,  0,  0, 33,  0, 36,
                                69,  0, 65,  0,  0,  0, 33,  0, 37, 
                                72,  0, 65,  0,  0,  0, 33,  0, 40,
                                69,  0, 65,  0,  0,  0, 33,  0, 37,
                                68,  0, 65,  0,  0,  0, 33,  0, 36,
                                67, 71, 65,  0,  0,  0, 33, 38, 35,
                                66,  0, 65,  0,  0,  0, 33,  0, 34,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int64)
    script_model = torch.jit.trace(model, input_data)
    script_model = torch.jit.script(model)
    model_path = f"./{args.game}_cat_transformer_bl{args.block_num}_ch{args.channel_num}.model"
    script_model.save(model_path)
    print(f"{model_path}にパラメータを保存")


if __name__ == "__main__":
    main()
