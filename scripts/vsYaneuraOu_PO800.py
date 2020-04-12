#! /usr/bin/env python3
import os
import sys

# Ayaneをインポート
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir + "/../../Ayane/source")
import shogi.Ayane as ayane

# その他必要なものをインポート
import time
import glob
from natsort import natsorted
from collections import defaultdict
import argparse
from calc_elo_rate import calc_elo_rate

parser = argparse.ArgumentParser()
parser.add_argument("--Threads", type=int, default=1)
parser.add_argument("--NodesLimit", type=int, default=100000)
parser.add_argument("--game_num", type=int, default=500)
args = parser.parse_args()

# 対局数(先後行うので偶数でなければならない)
assert args.game_num % 2 == 0

# 勝ち,負け,引き分けの結果を示す定数
WIN  = 0
DRAW = 1
LOSE = 2

# Ayaneにおける結果をここでの結果に変換する辞書
result_converter = { ayane.GameResult.BLACK_WIN: WIN,
                     ayane.GameResult.WHITE_WIN: LOSE,
                     ayane.GameResult.DRAW     : DRAW,
                     ayane.GameResult.MAX_MOVES: DRAW }


# インスタンス生成
server = ayane.AyaneruServer()

# サーバの設定
server.error_print = True
server.set_time_setting(f"byoyomi {10000000}")
server.moves_to_draw = 320

# YaneuraOuの設定
server.engines[1].set_engine_options({"USI_Ponder": "false",
                                      "Threads": args.Threads,
                                      "NodesLimit": args.NodesLimit,
                                      "USI_Hash": 1024,
                                      "BookMoves": 0,
                                      "NetworkDelay": 0,
                                      "NetworkDelay2": 0
                                      })
server.engines[1].connect(script_dir + "/../../YaneuraOu/bin/YaneuraOu-by-gcc")

# カレントディレクトリ内にある{prefix}_{step}.modelを評価する
curr_path = os.getcwd()
# ディレクトリ名が"/"で終わっていることの確認
if curr_path[-1] != "/":
    curr_path += "/"

# 結果を書き込むファイルを取得
f = open(curr_path + "result_PO800.txt", mode="a")
f.write(f"YaneuraOu Threads = {args.Threads} NodesLimit = {args.NodesLimit}\n")

# ディレクトリにある以下のprefixを持ったパラメータを用いて対局を行う
model_names = natsorted(glob.glob(curr_path + "*0.model"))
assert len(model_names) > 0

# パラメータを探索
for FPU_x1000 in range(-1000, 1001, 100):
    # Miacisを準備
    server.engines[0].set_engine_options({"random_turn": 320,
                                          "temperature_x1000": 10,
                                          "print_interval": 10000000,
                                          "USI_Hash": 4096,
                                          "search_limit": 800,
                                          "gpu_num": 1,
                                          "thread_num_per_gpu": 1,
                                          "search_batch_size": 4,
                                          "C_PUCT_x1000": 1500,
                                          "FPU_x1000": FPU_x1000,
                                          "model_name": model_names[-1]})
    scalar_or_categorical = "scalar" if "sca" in model_names[-1] else "categorical"
    server.engines[0].connect(f"{script_dir}/../src/cmake-build-release/Miacis_shogi_{scalar_or_categorical}")

    # 戦績を初期化
    total_num = [0, 0, 0]

    # 棋譜の集合を初期化
    sfens = defaultdict(int)

    # iが偶数のときMiacis先手
    for i in range(args.game_num):
        # 対局を実行
        server.game_start()
        while not server.game_result.is_gameover():
            time.sleep(1)

        # 重複を確認
        if sfens[server.sfen] > 0:
            # 同じ棋譜が2回生成された場合は記録しない
            print(f"\n重複:", server.sfen)
        else:
            # 結果を記録
            result = result_converter[server.game_result]
            total_num[result if not server.flip_turn else LOSE - result] += 1

        sfens[server.sfen] += 1

        # ここまでの結果を文字列化
        winning_rate = (total_num[WIN] + 0.5 * total_num[DRAW]) / sum(total_num)
        elo_rate = calc_elo_rate(winning_rate)
        result_str = f"FPU_x1000={FPU_x1000:5d} {total_num[WIN]:3d}勝 {total_num[DRAW]:3d}引き分け {total_num[LOSE]:3d}敗 勝率 {100 * winning_rate:4.1f}% 相対レート {elo_rate:6.1f}"

        sys.stdout.write("\033[2K\033[G")
        print(result_str, end="\n" if i == args.game_num - 1 else "")
        sys.stdout.flush()

        # 手番反転
        server.flip_turn = not server.flip_turn

    # ファイルに書き込み
    f.write(result_str + "\n")
    f.flush()


server.terminate()
