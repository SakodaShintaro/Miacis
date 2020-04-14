#!/usr/bin/env python3
import glob
import matplotlib.pyplot as plt
import japanize_matplotlib
from natsort import natsorted
from collections import defaultdict
import numpy as np

# 棋譜を分割する数
DIVIDE_NUM = 1

# 評価値を分類する幅
BIN_SIZE = 31
VALUE_COEFF = 5000
VALUE_WIDTH = 2 * VALUE_COEFF / BIN_SIZE

# ファイル一覧を取得
file_list = natsorted(glob.glob("./*.kifu"))
print(f"総棋譜数 = {len(file_list)}")

# 棋譜を分割する幅
width = len(file_list) // DIVIDE_NUM

# 初手,2手目のdict
first_dict = [defaultdict(int) for _ in range(DIVIDE_NUM)]
second_dict = [defaultdict(int) for _ in range(DIVIDE_NUM)]

# 手数
turn_nums = [list() for _ in range(DIVIDE_NUM)]

# eval_curveの統計
sum_reward = [[list() for _ in range(BIN_SIZE)] for _ in range(DIVIDE_NUM)]

# 棋譜を読み込んでデータ収集
for i, kifu_file_path in sorted(enumerate(file_list)):
    f = open(kifu_file_path).readlines()

    # この棋譜がどの分割に含まれるか
    index = i * DIVIDE_NUM // len(file_list)
    assert index < DIVIDE_NUM

    # 1手につき指し手と評価値の2行が出力されるため2で割ったものが手数
    turn_num = int(len(f) // 2)
    turn_nums[index].append(turn_num)

    # 極端に短い/長いものは出力してみる
    if turn_num <= 25 or 310 <= turn_num:
        print(kifu_file_path)

    # 最終結果の取得
    winner = ("先手" if (len(f) // 2 % 2) == 1 else "後手")
    reward = 1 if winner == "先手" else -1
    # 投了以外で終わっていたら出力
    if "投了" not in f[-1]:
        print(f[-1])

    # 初手の取得
    first = f[0].strip().split(" ")
    first_dict[index][first[1]] += 1

    # 2手目の取得
    second = f[2].strip().split(" ")
    second_dict[index][(first[1], second[1])] += 1

    # 評価値の取得
    for j, line in enumerate(f):
        if j % 2 == 0:
            continue
        line = line.strip().split(" ")
        value = float(line[2])
        value_index = min(int((value + VALUE_COEFF) // VALUE_WIDTH), BIN_SIZE - 1)
        sum_reward[index][value_index].append(reward)


# ターン数の統計情報を表示
for i in range(DIVIDE_NUM):
    print(f"i = {i}")
    print(f"棋譜の数 = {len(turn_nums[i])}")
    print(f"最短手数 = {np.min(turn_nums[i])}")
    print(f"最長手数 = {np.max(turn_nums[i])}")
    print(f"平均手数 = {np.mean(turn_nums[i]):.1f}")

# 指し手の情報
for i in range(DIVIDE_NUM):
    print(i, "first_dict")
    for k, v in sorted(first_dict[i].items(), key=lambda x:-x[1]):
        print(f"{k} {100 * v / len(turn_nums[i]):4.1f}")

    print(i, "second_dict")
    for k, v in sorted(second_dict[i].items(), key=lambda x:-x[1]):
        print(f"{k} {100 * v / len(turn_nums[i]):4.1f}")


# eval_curveの描画
x = [(j + 0.5) * 2 / BIN_SIZE - 1 for j in range(BIN_SIZE)]
for i in range(DIVIDE_NUM):
    y = [np.mean(sum_reward[i][j]) for j in range(BIN_SIZE)]
    plt.plot(x, y, label="段階"+str(i), marker=".")

plt.plot(x, x, linestyle="dashed", label="理論値")
plt.legend()

if DIVIDE_NUM > 1:
    plt.legend()
plt.xlabel("評価値")
plt.ylabel("平均報酬")
plt.savefig("eval_curve.png", bbox_inches="tight", pad_inches=0.05)
plt.clf()
