from torch.utils.data import Dataset
import torch
import cshogi
import numpy as np
from constant import *
import math


class HcpeDataSet(Dataset):
    def __init__(self, hcpe_file_path: str, is_valid: bool) -> None:
        super().__init__()
        self.hcpes_ = np.fromfile(hcpe_file_path, dtype=cshogi.HuffmanCodedPosAndEval)
        self.board_ = cshogi.Board()
        self.is_valid_ = is_valid

        # validデータのときは減らす
        if is_valid:
            self.hcpes_ = self.hcpes_[:len(self.hcpes_) // 20]

    def __len__(self):
        return len(self.hcpes_)

    def __getitem__(self, index):
        hcpe = self.hcpes_[index]
        self.board_.set_hcp(hcpe["hcp"])
        features = [0 for _ in range(SQUARE_NUM * INPUT_CHANNEL_NUM)]

        num_piece = len(PIECE_LIST)

        is_turn_black = (self.board_.turn == cshogi.BLACK)

        # 盤面
        for i, p in enumerate(PIECE_LIST):
            t = PIECE_LIST[i] if is_turn_black else PIECE_LIST[(i + num_piece // 2) % num_piece]
            for sq in cshogi.SQUARES:
                p = self.board_.piece(sq) if is_turn_black else self.board_.piece(SQUARE_NUM - 1 - sq)
                features[i * SQUARE_NUM + sq] = (1 if t == p else 0)

        # 持ち駒
        colors = [cshogi.BLACK, cshogi.WHITE] if is_turn_black else [cshogi.WHITE, cshogi.BLACK]
        for c in colors:
            for j in range(len(cshogi.HAND_PIECES)):
                for sq in cshogi.SQUARES:
                    features[i * SQUARE_NUM + sq] = self.board_.pieces_in_hand[c][j] / cshogi.MAX_PIECES_IN_HAND[j]
                i += 1

        x = torch.tensor(features)
        x = x.view([INPUT_CHANNEL_NUM, BOARD_SIZE, BOARD_SIZE])

        # Policy教師
        move16 = hcpe["bestMove16"]
        move = self.board_.move_from_move16(move16)
        policy_label = HcpeDataSet.make_output_label(move, self.board_.turn)
        policy_label = torch.tensor(policy_label)

        # value教師
        score_eval = 1 / (1 + math.exp(-hcpe["eval"] * 0.0013226)) * (MAX_SCORE - MIN_SCORE) + MIN_SCORE
        score_result = HcpeDataSet.make_value_label(hcpe["gameResult"], self.board_.turn)
        score = (score_eval + score_result) / 2 if not self.is_valid_ else score_result
        value_label = min(int((score - MIN_SCORE) // VALUE_WIDTH), int(BIN_SIZE - 1))
        value_label = torch.tensor(value_label)

        return x, policy_label, value_label

    @staticmethod
    def make_output_label(move, color):
        move_to = cshogi.move_to(move)
        move_from = cshogi.move_from(move)

        if move_from >= SQUARE_NUM:
            move_from = None

        # 白の場合盤を回転
        if color == cshogi.WHITE:
            move_to = SQUARE_NUM - 1 - move_to
            if move_from is not None:
                move_from = SQUARE_NUM - 1 - move_from

        # move direction
        if move_from is not None:
            to_y, to_x = divmod(move_to, 9)
            from_y, from_x = divmod(move_from, 9)
            dir_x = to_x - from_x
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
            move_direction = len(MOVE_DIRECTION) + cshogi.move_drop_hand_piece(move) - 1

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
