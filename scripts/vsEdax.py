#! /usr/bin/env python3
import subprocess
import os
import sys
import glob
from natsort import natsorted
import argparse
from calc_elo_rate import calc_elo_rate


class EdaxManager:
    # Edaxの盤面表示の最後の行（正確にはこの行の後に改行だけの行が出力される）
    target_str = "  A B C D E F G H            WHITE            A  B  C  D  E  F  G  H\n"

    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.proc = subprocess.Popen([f"{script_dir}/../../Edax/edax-4.4"],
                                     cwd=f"{script_dir}/../../Edax",
                                     encoding="UTF-8",
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)

    def read_lines(self):
        curr_line = ""
        best_move = None
        game_over = False
        win_color = None

        while True:
            pre_line = curr_line
            curr_line = self.proc.stdout.readline()
            # print(curr_line, end="")

            if "Edax plays" in curr_line:
                best_move = curr_line[-3:-1]
            if "Game over" in curr_line:
                game_over = True
            if "won" in curr_line:
                elements = curr_line.strip().split()
                win_color = elements[elements.index("won") - 1]

            pre_match = (pre_line == EdaxManager.target_str)
            curr_match = (curr_line == "\n")

            if pre_match and curr_match:
                break

        return best_move, game_over, win_color

    def send_message(self, message):
        self.proc.stdin.write(message + "\n")
        self.proc.stdin.flush()
        return self.read_lines()


class MiacisManager:
    def __init__(self, scalar_or_categorical_str):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.proc = subprocess.Popen(
            [f"{script_dir}/../src/cmake-build-release/Miacis_othello_{scalar_or_categorical_str}"],
            cwd=f"{script_dir}/../src/cmake-build-release",
            encoding="UTF-8",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)
        # 最初はCUDAの利用判定が一行入るので読み飛ばす
        self.proc.stdout.readline()

    def send(self, message):
        self.proc.stdin.write(message + "\n")
        self.proc.stdin.flush()

    def send_option(self, name, value):
        self.send(f"set {name} {value}")

    def send_init(self):
        self.send("init")

    def send_play(self, move):
        self.send("play " + move)

    def send_go(self):
        self.send("go")
        while True:
            line = self.proc.stdout.readline()
            if "best_move" in line:
                line = line.strip().split()
                return line[1]


def main():
    result_dict = {
        "Black": 0,
        None: 1,
        "White": 2
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=6)
    parser.add_argument("--game_num", type=int, default=500)
    parser.add_argument("--search_limit", type=int, default=800)
    parser.add_argument("--init_model_step", type=int, default=0)
    parser.add_argument("--search_batch_size", type=int, default=4)
    parser.add_argument("--temperature_x1000", type=int, default=75)
    parser.add_argument("--exp_search", action="store_true")
    parser.add_argument("--mcts_net", action="store_true")
    parser.add_argument("--proposed_model", action="store_true")
    parser.add_argument("--stacked_lstm", action="store_true")
    parser.add_argument("--use_readout_only", action="store_true")
    args = parser.parse_args()

    # カレントディレクトリ内にある{prefix}_{step}.modelを評価する
    curr_path = os.getcwd()
    # ディレクトリ名が"/"で終わっていることの確認
    if curr_path[-1] != "/":
        curr_path += "/"

    # ディレクトリにある以下のprefixを持ったパラメータを用いて対局を行う
    model_names = natsorted(glob.glob(curr_path + "*0.model"))

    # プロセスmanagerを準備
    edax_manager = EdaxManager()
    edax_manager.read_lines()
    edax_manager.send_message(f"set level {args.level}")

    miacis_manager = MiacisManager("categorical" if "cat" in model_names[0] else "scalar")
    miacis_manager.send_option("search_limit", args.search_limit)
    miacis_manager.send_option("byoyomi_margin", 10000000)
    miacis_manager.send_option("search_batch_size", args.search_batch_size)
    miacis_manager.send_option("temperature_x1000", args.temperature_x1000)
    miacis_manager.send_option("gpu_num", 1)
    if args.exp_search:
        miacis_manager.send_option("Q_coeff_x1000", 1000)
        miacis_manager.send_option("P_coeff_x1000", 0)
    miacis_manager.send_option("thread_num_per_gpu", 1)
    miacis_manager.send_option("random_turn", 30)

    if args.mcts_net:
        miacis_manager.send_option("use_mcts_net", "true")
    elif args.proposed_model:
        miacis_manager.send_option("use_proposed_model", "true")
    elif args.stacked_lstm:
        miacis_manager.send_option("use_stacked_lstm", "true")

    if args.use_readout_only:
        miacis_manager.send_option("use_readout_only", "true")

    # 結果を書き込むファイルを取得
    f = open(curr_path + "result.txt", mode="a")
    f.write(f"level = {args.level}, "
            f"search_limit = {args.search_limit}, "
            f"search_batch_size = {args.search_batch_size}, "
            f"temperature_x1000 = {args.temperature_x1000}\n")

    for model_name in model_names:
        # 最後に出てくるアンダーバーから.modelの直前までにステップ数が記録されているという前提
        step = int(model_name[model_name.rfind("_") + 1:model_name.find(".model")])
        if step < args.init_model_step:
            continue
        miacis_manager.send_option("model_name", model_name)

        # 対局を文字列化して同一性を判定するための集合
        moves_set = set()

        # 勝敗の結果
        total_result = [0, 0, 0]

        # 全く同じ対局が行われた回数
        repetition_num = 0

        game_num = 0
        while game_num < args.game_num:
            # 局面を初期化
            miacis_manager.send_init()
            best_move, game_over, win_color = edax_manager.send_message("init")

            # 行動のリストを初期化
            moves = list()

            is_miacis_white = (game_num % 2 == 1)
            for turn_num in range(1000):
                if (turn_num + is_miacis_white) % 2 == 0:
                    # Miacisのターン
                    best_move = miacis_manager.send_go()
                    _, game_over, win_color = edax_manager.send_message("play " + best_move)
                else:
                    # Edaxのターン
                    best_move, game_over, win_color = edax_manager.send_message("go")
                    miacis_manager.send_play(best_move)
                moves.append(best_move)
                if game_over:
                    break

            moves = tuple(moves)

            if moves in moves_set:
                # 以前の対局と重複があったということなので飛ばす
                repetition_num += 1
                continue

            game_num += 1
            moves_set.add(moves)

            result = result_dict[win_color]
            if is_miacis_white:
                result = 2 - result

            total_result[result] += 1

            winning_rate = (total_result[0] + 0.5 * total_result[1]) / sum(total_result)
            elo_rate = calc_elo_rate(winning_rate)
            result_str = f"{step:7d}ステップ {total_result[0]:3d}勝 {total_result[1]:3d}引き分け {total_result[2]:3d}敗 勝率 {100 * winning_rate:4.1f}% 相対レート {elo_rate:6.1f} 重複 {repetition_num:3d}"

            sys.stdout.write("\033[2K\033[G")
            print(result_str, end="\n" if game_num == args.game_num else "")
            sys.stdout.flush()
        f.write(result_str + "\n")


if __name__ == '__main__':
    main()
