#!/usr/bin/env python3
import subprocess


class EdaxManager:
    # Edaxの盤面表示の最後の行（正確にはこの行の後に改行だけの行が出力される）
    target_str = "  A B C D E F G H            WHITE            A  B  C  D  E  F  G  H\n"

    def __init__(self):
        self.proc = subprocess.Popen(["./edax-4.4"],
                                     cwd="../../edax-linux",
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
            print(curr_line, end="")

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
        self.proc = subprocess.Popen(["./Miacis_othello_categorical"],
                                     cwd="../src/cmake-build-release/",
                                     encoding="UTF-8",
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)
        # 最初はCUDAの利用判定が一行入るので読み飛ばす
        self.proc.stdout.readline()

    def send(self, message):
        self.proc.stdin.write(message + "\n")
        self.proc.stdin.flush()

    def send_option(self, name, value):
        self.send("set " + str(name) + " " + str(value))

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

    miacis_manager.send_option("search_limit", 800)
    miacis_manager.send_option("byoyomi_margin", 10000000)
    miacis_manager.send_option("search_batch_size", 1)
    miacis_manager.send_option("thread_num", 1)

    result_dict = {
        "Black": 0,
        None: 1,
        "White": 2
    }

    total_result = [0, 0, 0]

    # 最初の読み込み
    edax_manager.read_lines()

    messages = [
        "set verbose 1",
        "set level 1",
        "set book-randomness 10",
    ]

    # 設定の送信
    for msg in messages:
        edax_manager.send_message(msg)

    for game_num in range(10):
        # 局面を初期化
        miacis_manager.send_init()
        best_move, game_over, win_color = edax_manager.send_message("init")

        is_miacis_white = (game_num % 2 == 1)
        for turn_num in range(1000):
            if (turn_num + is_miacis_white) % 2 == 0:
                # Miacisのターン
                best_move = miacis_manager.send_go()
                best_move, game_over, win_color = edax_manager.send_message("play " + best_move)
            else:
                # Edaxのターン
                best_move, game_over, win_color = edax_manager.send_message("go")
                miacis_manager.send_play(best_move)
            if game_over:
                break

        result = result_dict[win_color]
        if is_miacis_white:
            result = 2 - result

        total_result[result] += 1
        print(total_result)


if __name__ == '__main__':
    main()
