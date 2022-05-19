from torch.utils.data import Dataset
import torch
import cshogi
import numpy as np
from constant import *
import math
import random


class RandomMoveDataSet(Dataset):
    def __init__(self, is_valid: bool) -> None:
        super().__init__()
        self.board_ = cshogi.Board()
        self.is_valid_ = is_valid

    def __len__(self):
        return 1000

    def __getitem__(self, index):
        moves = list(self.board_.legal_moves)

        curr_turn = self.board_.turn

        x1 = RandomMoveDataSet.make_input(self.board_)
        move = random.choice(moves)
        self.board_.push_usi(cshogi.move_to_usi(move))
        x2 = RandomMoveDataSet.make_input(self.board_)

        if self.board_.is_game_over():
            self.board_.reset()

        # Policy教師
        policy_label = RandomMoveDataSet.make_output_label(move, curr_turn)
        policy_label = torch.tensor(policy_label)

        return x1, x2, policy_label

    @staticmethod
    def make_input(board):
        features = [0 for _ in range(SQUARE_NUM * INPUT_CHANNEL_NUM)]

        num_piece = len(PIECE_LIST)

        is_turn_black = (board.turn == cshogi.BLACK)

        # 盤面
        for i, p in enumerate(PIECE_LIST):
            t = PIECE_LIST[i] if is_turn_black else PIECE_LIST[(i + num_piece // 2) % num_piece]
            for sq in cshogi.SQUARES:
                p = board.piece(sq) if is_turn_black else board.piece(SQUARE_NUM - 1 - sq)
                features[i * SQUARE_NUM + sq] = (1 if t == p else 0)

        # 持ち駒
        colors = [cshogi.BLACK, cshogi.WHITE] if is_turn_black else [cshogi.WHITE, cshogi.BLACK]
        for c in colors:
            for j in range(len(cshogi.HAND_PIECES)):
                i += 1
                for sq in cshogi.SQUARES:
                    features[i * SQUARE_NUM + sq] = board.pieces_in_hand[c][j] / cshogi.MAX_PIECES_IN_HAND[j]

        x = torch.tensor(features)
        x = x.view([INPUT_CHANNEL_NUM, BOARD_SIZE, BOARD_SIZE])
        return x

    @staticmethod
    def make_output_label(move, color):
        move_to = cshogi.move_to(move)
        move_from = cshogi.move_from(move)

        drop_piece = None
        if move_from >= SQUARE_NUM:
            drop_piece = move_from - SQUARE_NUM
            move_from = None

        # 白の場合盤を回転
        if color == cshogi.WHITE:
            move_to = SQUARE_NUM - 1 - move_to
            if move_from is not None:
                move_from = SQUARE_NUM - 1 - move_from

        # move direction
        if move_from is not None:
            to_x, to_y = divmod(move_to, 9)
            from_x, from_y = divmod(move_from, 9)
            dir_x = -(to_x - from_x)
            dir_y = to_y - from_y
            if dir_y < 0 and dir_x == 0:
                move_direction = UP
            elif dir_y == -2 and dir_x == -1:
                move_direction = UP2_LEFT
            elif dir_y == -2 and dir_x == 1:
                move_direction = UP2_RIGHT
            elif dir_y < 0 and dir_x < 0:
                move_direction = UP_LEFT
            elif dir_y < 0 and dir_x > 0:
                move_direction = UP_RIGHT
            elif dir_y == 0 and dir_x < 0:
                move_direction = LEFT
            elif dir_y == 0 and dir_x > 0:
                move_direction = RIGHT
            elif dir_y > 0 and dir_x == 0:
                move_direction = DOWN
            elif dir_y > 0 and dir_x < 0:
                move_direction = DOWN_LEFT
            elif dir_y > 0 and dir_x > 0:
                move_direction = DOWN_RIGHT

            # promote
            if cshogi.move_is_promotion(move):
                move_direction = MOVE_DIRECTION_PROMOTED[move_direction]
        else:
            # 持ち駒
            move_direction = len(MOVE_DIRECTION) + drop_piece

        move_label = 9 * 9 * move_direction + move_to

        return move_label

    @staticmethod
    def make_value_label(result, color):
        if color == cshogi.BLACK:
            if result == cshogi.DRAW:
                return (MIN_SCORE + MAX_SCORE) / 2
            elif result == cshogi.BLACK_WIN:
                return MAX_SCORE
            else:
                return MIN_SCORE
        else:
            if result == cshogi.DRAW:
                return (MIN_SCORE + MAX_SCORE) / 2
            elif result == cshogi.BLACK_WIN:
                return MIN_SCORE
            else:
                return MAX_SCORE
