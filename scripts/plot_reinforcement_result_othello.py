#!/usr/bin/env python3
import matplotlib.pyplot as plt
import os
import japanize_matplotlib
import re
import argparse

markers = [".", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "*", "h", "H", "+", "x", "D"]

STEP = 0
ELO_RATE = 1
ELEMENT_NUM = 2

# ディレクトリの名前をコマンドライン引数として受け取る
parser = argparse.ArgumentParser()
parser.add_argument("-dirs", type=(lambda x: x.split()))
parser.add_argument("--labels", type=(lambda x: x.split("%")), default=None)
args = parser.parse_args()
if args.labels is None:
    args.labels = [""]

assert len(args.dirs) == len(args.labels)

# 3次元
# 1次元目:各学習
# 2次元目:項目, TIMEかSTEPかなど
# 3次元目:各学習の各項目における各ステップの値
all_data = list()

# データの取得
for dir_name in args.dirs:
    # このディレクトリのデータ
    data = [list() for _ in range(ELEMENT_NUM)]

    # 対局結果はresult.txtにある
    result_file_name = dir_name + "/result.txt"
    if os.path.exists(result_file_name):
        for line in open(result_file_name):
            # 空白区切りで"相対レート"という要素の次にレートが記録されていることを前提とする
            elements = line.strip().split()
            if "ステップ" not in elements[0]:
                continue
            data[STEP].append(int(elements[0][:elements[0].index("ステップ")]))
            if "相対レート" in elements:
                data[ELO_RATE].append(float(elements[elements.index("相対レート") + 1]))
    else:
        print("There is not a such file : ", result_file_name)

    all_data.append(data)

# グラフの描画
plt.xlabel("学習ステップ数")
plt.ylabel("相対Eloレート")

texts = list()
for k, data in enumerate(all_data):
    plt.plot(data[STEP], data[ELO_RATE], label=args.labels[k], marker=markers[k])
    texts.append(plt.text(data[STEP][-1] * 1.02, data[ELO_RATE][-1], args.labels[k], color=plt.get_cmap("tab10")(k)))
texts.sort(key=lambda text: text.get_position()[1])
pre_y = -float("inf")
margin = (plt.ylim()[1] - plt.ylim()[0]) / 30
for text in texts:
    x, y = text.get_position()
    y = max(y - margin / 3, pre_y + margin)
    pre_y = y
    text.set_position((x, y))

e_label = "Edax(Level 5)"
plt.plot(all_data[0][STEP], [0 for _ in range(len(all_data[0][STEP]))], linestyle="dashed", label=e_label)
plt.text(all_data[0][STEP][0], 5, e_label, color=plt.get_cmap("tab10")(len(all_data)))

# 目盛り表示量が多いので間引く
# locs, labels = plt.xticks()
# plt.xticks(locs[::2])

if len(all_data) > 1:
    x_min, x_max = plt.xlim()
    plt.xlim((x_min, x_max * 1.175))

    # plt.legend()
plt.savefig("graph.png", bbox_inches='tight', pad_inches=0.05)
plt.clf()
