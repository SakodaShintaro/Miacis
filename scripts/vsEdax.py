#!/usr/bin/env python3
import subprocess


class EdaxManager:
    # Edaxの盤面表示の最後の行（正確にはこの行の後に改行だけの行が出力される）
    target_str = "  A B C D E F G H            WHITE            A  B  C  D  E  F  G  H\n"

    def __init__(self):
        self.proc = subprocess.Popen(["/home/sakoda/edax-linux/edax-4.4"],
                                     cwd="/home/sakoda/edax-linux",
                                     encoding="UTF-8",
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)

    def read_lines(self):
        pre_line = ""
        curr_line = ""
        best_move = None
        game_over = False

        while True:
            pre_line = curr_line
            curr_line = self.proc.stdout.readline()
            print(curr_line, end="")

            if "Edax plays" in curr_line:
                best_move = curr_line[-3:-1]
            if "Game over" in curr_line:
                game_over = True

            pre_match = (pre_line == EdaxManager.target_str)
            curr_match = (curr_line == "\n")

            if pre_match and curr_match:
                break

        return best_move, game_over

    def send_message(self, message):
        self.proc.stdin.write(message + "\n")
        self.proc.stdin.flush()
        print(message)
        return self.read_lines()


class MiacisManager:
    def __init__(self):
        self.proc = subprocess.Popen(["../src/cmake-build-release/Miacis_othello_categorical"],
                                     cwd="../src/cmake-build-release/",
                                     encoding="UTF-8",
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)

    def read_lines(self):
        best_move = None
        line = self.proc.stdout.readline()
        line.strip().split()
        if len(line) > 1:
            best_move = line[1]
        return best_move

    def send(self, message):
        self.proc.stdin.write(message + "\n")
        self.proc.stdin.flush()
        print(message)


def main():
    messages = [
        "help",
        "set verbose 1",
        "set level 13",
        "set book-randomness 10",
    ]

    edax_manager = EdaxManager()

    # 最初の読み込み
    edax_manager.read_lines()

    # 設定の送信
    for msg in messages:
        edax_manager.send_message(msg)

    # 対局の実行
    while True:
        best_move, game_over = edax_manager.send_message("go")
        print("best_move =", best_move)
        if game_over:
            break


if __name__ == '__main__':
    main()
