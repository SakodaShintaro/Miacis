#!/usr/bin/env python3
import subprocess
import os


class EdaxManager:
    # Edaxの盤面表示の最後の行（正確にはこの行の後に改行だけの行が出力される）
    target_str = "  A B C D E F G H            WHITE            A  B  C  D  E  F  G  H\n"

    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.proc = subprocess.Popen([f"{script_dir}/../../edax-linux/edax-4.4"],
                                     cwd=f"{script_dir}/../../edax-linux",
                                     encoding="UTF-8",
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)

    def read_lines(self):
        pre_line = ""
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
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.proc = subprocess.Popen([f"{script_dir}/../src/cmake-build-release/Miacis_othello_categorical"],
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
    edax_manager = EdaxManager()
    miacis_manager = MiacisManager()

    miacis_manager.send_option("search_limit", 100)
    miacis_manager.send_option("byoyomi_margin", 10000000)
    miacis_manager.send_option("search_batch_size", 1)
    miacis_manager.send_option("thread_num", 1)
    miacis_manager.send_option("random_turn", 1000)

    result_dict = {
        "Black": 0,
        None: 1,
        "White": 2
    }

    moves_set = set()

    total_result = [0, 0, 0]

    # 最初の読み込み
    edax_manager.read_lines()

    messages = [
        "set verbose 1",
        "set level 1",
    ]

    # 設定の送信
    for msg in messages:
        edax_manager.send_message(msg)

    game_num = 0
    while game_num < 100:
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
            print("重複発生")
            continue

        print(moves)

        game_num += 1
        moves_set.add(moves)

        result = result_dict[win_color]
        if is_miacis_white:
            result = 2 - result

        total_result[result] += 1
        print(total_result)


if __name__ == '__main__':
    main()
