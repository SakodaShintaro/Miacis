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
parser.add_argument("--time1", type=int, default=1000)
parser.add_argument("--time2", type=int, default=400)
parser.add_argument("--Threads", type=int, default=4)
parser.add_argument("--NodesLimit", type=int, default=0)
parser.add_argument("--game_num", type=int, default=500)
parser.add_argument("--init_model_step", type=int, default=0)
parser.add_argument("--reverse", action="store_true")
parser.add_argument("--option", type=str, default=None)
parser.add_argument("--parameters", type=(lambda x: list(map(int, x.split()))))
parser.add_argument("--Suisho", action="store_true")
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
server.set_time_setting(f"byoyomi1p {args.time1} byoyomi2p {args.time2}")
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
if args.Suisho:
    server.engines[1].connect(script_dir + "/../../Suisho/bin/YaneuraOu-by-gcc")
else:
    server.engines[1].connect(script_dir + "/../../YaneuraOu/bin/YaneuraOu-by-gcc")

# カレントディレクトリ内にある{prefix}_{step}.modelを評価する
curr_path = os.getcwd()
# ディレクトリ名が"/"で終わっていることの確認
if curr_path[-1] != "/":
    curr_path += "/"

# 結果を書き込むファイルを取得
f = open(curr_path + "result.txt", mode="a")
f.write(f"Miacis time = {args.time1}, YaneuraOu time = {args.time2}, YaneuraOu Threads = {args.Threads}\n")

# ディレクトリにある以下のprefixを持ったパラメータを用いて対局を行う
model_names = natsorted(glob.glob(curr_path + "*0.model"))
assert len(model_names) > 0

if args.reverse:
    model_names = model_names[::-1]

if args.option is None:
    # 各ステップの勝率を計測
    for model_name in model_names:
        # 最後に出てくるアンダーバーから.modelの直前までにステップ数が記録されているという前提
        step = int(model_name[model_name.rfind("_") + 1:model_name.find(".model")])

        # args.init_model_stepより小さいものは調べない
        if step < args.init_model_step:
            continue

        # Miacisを準備
        server.engines[0].set_engine_options({"random_turn": 30,
                                              "print_interval": 10000000,
                                              "USI_Hash": 4096,
                                              "model_name": model_name})
        scalar_or_categorical = "scalar" if "sca" in model_name else "categorical"
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
            result_str = f"{step:7d}ステップ {total_num[WIN]:3d}勝 {total_num[DRAW]:3d}引き分け {total_num[LOSE]:3d}敗 勝率 {100 * winning_rate:4.1f}% 相対レート {elo_rate:6.1f}"

            sys.stdout.write("\033[2K\033[G")
            print(result_str, end="\n" if i == args.game_num - 1 else "")
            sys.stdout.flush()

            # 手番反転
            server.flip_turn = not server.flip_turn

        # ファイルに書き込み
        f.write(result_str + "\n")
        f.flush()
else:
    # パラメータを探索
    for parameter in args.parameters:
        # Miacisを準備
        server.engines[0].set_engine_options({"random_turn": 30,
                                              "print_interval": 10000000,
                                              "USI_Hash": 4096,
                                              args.option: parameter,
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
            result_str = f"{args.option}={parameter:7d} {total_num[WIN]:3d}勝 {total_num[DRAW]:3d}引き分け {total_num[LOSE]:3d}敗 勝率 {100 * winning_rate:4.1f}% 相対レート {elo_rate:6.1f}"

            sys.stdout.write("\033[2K\033[G")
            print(result_str, end="\n" if i == args.game_num - 1 else "")
            sys.stdout.flush()

            # 手番反転
            server.flip_turn = not server.flip_turn

        # ファイルに書き込み
        f.write(result_str + "\n")
        f.flush()


server.terminate()
