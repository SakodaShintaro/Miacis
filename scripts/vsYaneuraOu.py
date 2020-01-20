#! /usr/bin/env python3

import os
import sys

# カレントディレクトリ内にある{prefix}_{step}.modelを評価する
curr_path = os.getcwd()
# ディレクトリ名が"/"で終わっていることの確認
if curr_path[-1] != "/":
    curr_path += "/"

# Ayaneをインポート
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir + "/../Ayane/source")
import shogi.Ayane as ayane

# その他必要なものをインポート
import time
import math
import glob
from natsort import natsorted
from collections import defaultdict

# 対局数(先後行うので偶数でなければならない)
game_num = 500
assert game_num % 2 == 0

# 持ち時間
time_yane   = 100
time_Miacis = 1000
time_max = max(time_yane, time_Miacis)

# ディレクトリ内のパラメータのうち指定したステップ以降のモデルを対局させる
min_step = 0

# 勝ち,負け,引き分けの結果を示す定数
WIN  = 0
DRAW = 1
LOSE = 2

# Ayaneにおける結果をここでの結果に変換する辞書
result_converter = { ayane.GameResult.BLACK_WIN: WIN,
                     ayane.GameResult.WHITE_WIN: LOSE,
                     ayane.GameResult.DRAW     : DRAW,
                     ayane.GameResult.MAX_MOVES: DRAW }


# 勝率からelo_rateを計算する関数
def calc_elo_rate(winning_rate):
    assert 0 <= winning_rate <= 1
    if winning_rate == 1.0:
        return 10000.0
    elif winning_rate == 0.0:
        return -10000.0
    else:
        return 400 * math.log10(winning_rate / (1 - winning_rate))


# インスタンス生成
server = ayane.AyaneruServer()

# サーバの設定
server.error_print = True
server.set_time_setting(f"byoyomi {time_max}")
server.moves_to_draw = 256

# YaneuraOuの設定
server.engines[1].set_engine_options({"USI_Ponder": "false",
                                      "Threads": 1,
                                      "BookMoves": 0,
                                      "NetworkDelay": time_max - time_yane,
                                      "NetworkDelay2": time_max - time_yane
                                      })
server.engines[1].connect(script_dir + "/../YaneuraOu/bin/YaneuraOu-by-gcc")

# 結果を書き込むファイルを取得
f = open(curr_path + "result.txt", mode="a")

# ディレクトリにある以下のprefixを持ったパラメータを用いて対局を行う
model_names = natsorted(glob.glob(curr_path + "*0.model"))

for model_name in model_names:
    # 最後に出てくるアンダーバーから.modelの直前までにステップ数が記録されているという前提
    step = int(model_name[model_name.rfind("_") + 1:model_name.find(".model")])

    # min_stepより小さいものは調べない
    if step < min_step:
        continue

    # Miacisを準備
    server.engines[0].set_engine_options({"byoyomi_margin": time_max - time_Miacis,
                                          "random_turn": 30,
                                          "print_interval": 10000000,
                                          "USI_Hash": 4096,
                                          "C_PUCT_x1000": 2500,
                                          "model_name": model_name})
    scalar_or_categorical = "scalar" if "sca" in model_name else "categorical"
    server.engines[0].connect(f"{script_dir}/../src/cmake-build-release/Miacis_shogi_{scalar_or_categorical}")

    # 戦績を初期化
    total_num = [0, 0, 0]

    # 棋譜の集合を初期化
    sfens = defaultdict(int)

    # iが偶数のときMiacis先手
    for i in range(game_num):
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
        print(result_str, end="\n" if i == game_num - 1 else "")
        sys.stdout.flush()

        # 手番反転
        server.flip_turn = not server.flip_turn

    # ファイルに書き込み
    f.write(result_str + "\n")
    f.flush()
server.terminate()
