#!/usr/bin/env python3
import glob
import codecs
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from natsort import natsorted
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dir", type=str)
args = parser.parse_args()

# もし序盤が弱い→序盤から悪くしてそのまま負ける
# もし終盤が弱い→序盤・中盤は良いのに終盤で負ける
turns = list()

BIN_SIZE = 31
BIN_WIDTH = 2 / BIN_SIZE
result_points = [list() for _ in range(BIN_SIZE)]

PHASE_NUM = 3
result_points_each_phase = [[list() for _ in range(BIN_SIZE)] for i in range(PHASE_NUM)]

total_result_for_miacis = [0, 0, 0]

file_names = natsorted(glob.glob(f"{args.dir}/*.kif"))
for file_name in file_names:
    f = codecs.open(file_name, 'r', 'shift_jis')
    date = f.readline().strip()
    startpos = f.readline().strip()
    black = f.readline().strip()
    white = f.readline().strip()
    label = f.readline().strip()

    is_miacis_black = "Miacis" in black

    miacis_scores = list()

    result = None

    while True:
        # 指し手が記述されている行を読み込み
        line1 = f.readline().strip()
        elements1 = line1.split()

        # 評価値が記述されている行を読み込み
        line2 = f.readline().strip()
        elements2 = line2.split()

        # 指し手を取得
        turn = int(elements1[0])
        move = elements1[1]

        # 同* という行動は"同 *"と記録されるため分割されてしまう
        if move == "同":
            move += elements1[2]

        if move == "投了":
            # print(turn, move, end=" ")
            turns.append(turn - 1)

            # 読み筋が記録されている場合があるのでコメントだったら読み込み直す
            if line2[0:2] == "**":
                line2 = f.readline().strip()

            # 勝敗を解釈
            if "先手の勝ち" in line2:
                # print(f"勝者:{black}")
                result = 1
            elif "後手の勝ち" in line2:
                # print(f"勝者:{white}")
                result = -1
            else:
                print(line2)
                assert False
            break
        elif move == "入玉宣言":
            print(file_name)
            print(turn, move, end=" ")
            turns.append(turn - 1)

            # 読み筋が記録されている場合があるのでコメントだったら読み込み直す
            if line2[0:2] == "**":
                line2 = f.readline().strip()

            # 勝敗を解釈
            if turn % 2 == 1:
                print(f"勝者:{black}")
                result = 1
            else:
                print(f"勝者:{white}")
                result = -1
            break
        elif move == "持将棋" or move == "千日手":
            print(file_name, move)
            turns.append(turn - 1)

            # 読み筋が記録されている場合があるのでコメントだったら読み込み直す
            if line2[0:2] == "**":
                line2 = f.readline().strip()

            # 勝敗を解釈
            result = 0
            break

        # 評価値を取得
        score_index = elements2.index("評価値") + 1
        score = elements2[score_index]

        # 詰みだとスペース区切りで次に手数が記録されるため分割されている
        if "詰" in score:
            score += elements2[score_index + 1]

        # print(turn, move, score)
        if (turn % 2 == 1 and is_miacis_black) or (turn % 2 == 0 and not is_miacis_black):
            miacis_scores.append(float(score) / 5000)

    result_for_miacis = result if is_miacis_black else -result
    total_result_for_miacis[int(1 - result_for_miacis)] += 1

    for i, score in enumerate(miacis_scores):
        index = min(int((score + 1) // BIN_WIDTH), BIN_SIZE - 1)
        result_points[index].append(result)

        phase = min(i * PHASE_NUM // len(miacis_scores), PHASE_NUM - 1)
        result_points_each_phase[phase][index].append(result)
print(f"対局数 {len(turns)}")
print(f"最小手数 {np.min(turns)}")
print(f"最大手数 {np.max(turns)}")
print(f"平均手数 {np.mean(turns)}")
print(f"標準偏差 {np.std(turns)}")

print("Miacisから見た勝敗")
print(f"{total_result_for_miacis[0]}勝 {total_result_for_miacis[1]}引き分け {total_result_for_miacis[2]}敗")

x = [-1 + BIN_WIDTH * (i + 0.5) for i in range(BIN_SIZE)]
y = list()
y_each_phase = [list() for _ in range(PHASE_NUM)]
for i in range(BIN_SIZE):
    y.append(np.mean(result_points[i]))
    for p in range(PHASE_NUM):
        y_each_phase[p].append(np.mean(result_points_each_phase[p][i]))

plt.plot(x, y, marker=".", label="Miacisの探索結果")
plt.plot(x, x, linestyle="dashed", label="理論値")
plt.legend()

plt.xlabel("評価値(探索結果)")
plt.ylabel("平均報酬")
plt.savefig(f"{args.dir}/evaluation_curve.png", bbox_inches="tight", pad_inches=0.05)
