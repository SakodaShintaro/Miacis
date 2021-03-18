#!/usr/bin/env python3
import matplotlib.pyplot as plt
import os
import japanize_matplotlib
import re
import argparse

markers = [".", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "*", "h", "H", "+", "x", "D"]

TIME = 0
STEP = 1
POLICY_LOSS = 2
VALUE_LOSS = 3
ELO_RATE = 4
ELEMENT_NUM = 5

# ディレクトリの名前をコマンドライン引数として受け取る
parser = argparse.ArgumentParser()
parser.add_argument("-dirs", type=(lambda x: x.split()))
parser.add_argument("--labels", type=(lambda x: x.split()), default=None)
args = parser.parse_args()
if args.labels is None:
    args.labels = [""]

assert len(args.dirs) == len(args.labels)

# 3次元
# 1次元目:各学習
# 2次元目:項目, TIMEかSTEPかなど
# 3次元目:各学習の各項目における各ステップの値
all_data = list()
label = None

# データの取得
for dir_name in args.dirs:
    # このディレクトリのデータ
    data = [list() for _ in range(ELEMENT_NUM)]

    # まず損失のデータを取得する
    loss_file_name = dir_name + "/reinforcement_valid_log.txt"
    if not os.path.exists(loss_file_name):
        print("There is not a such file : ", loss_file_name)
        break
    f = open(loss_file_name)

    # 最初の1行にはラベルがあることを前提とする
    labels = [x for x in f.readline().strip().split("\t")]

    # 対局結果だけは損失のラベルにないので追加する
    labels.append("Elo_rate")

    # 最初だけlabelを保存
    if not label:
        label = labels

    for line in f:
        line = line.strip().split("\t")
        for i in range(len(line)):
            try:
                data[i].append(float(line[i]))
            except:
                e = line[i].split(":")
                hour = float(e[0]) + float(e[1]) / 60 + float(e[2]) / 3600
                data[i].append(hour)

    all_data.append(data)

# timeという名前にしているが時間で換算した方がわかりやすいので名前を変える
label[TIME] = "hour"

# グラフの描画
for i in [STEP, TIME]:  # x軸
    for j in [POLICY_LOSS, VALUE_LOSS]:  # y軸
        plt.xlabel(label[i])
        plt.ylabel(label[j])

        texts = list()
        for k, data in enumerate(all_data):
            d = len(data[i]) // len(data[j])
            plt.plot(data[i][d - 1::d], data[j], label=args.labels[k], marker=markers[k])
            if len(all_data) > 1:
                texts.append(plt.text(data[i][-1] * 1.01, data[j][-1], args.labels[k], color=plt.get_cmap("tab10")(k)))
        texts.sort(key=lambda text: text.get_position()[1])
        pre_y = -float("inf")
        margin = (plt.ylim()[1] - plt.ylim()[0]) / 30
        for text in texts:
            x, y = text.get_position()
            y = max(y - margin / 3, pre_y + margin)
            pre_y = y
            text.set_position((x, y))

        if j == ELO_RATE:
            e_label = "YO/Kristallweizen(4Thread, 0.2sec)"
            plt.plot(all_data[0][i], [0 for _ in range(len(all_data[0][i]))], linestyle="dashed", label=e_label)
            plt.text(all_data[0][i][0], 10, e_label, color=plt.get_cmap("tab10")(len(all_data)))

        # 目盛り表示量が多いので間引く
        # locs, labels = plt.xticks()
        # plt.xticks(locs[::2])

        if len(all_data) > 1:
            x_min, x_max = plt.xlim()
            plt.xlim((x_min, x_max * 1.175))

            # plt.legend()
        plt.savefig(label[i] + "-" + label[j] + ".png", bbox_inches='tight', pad_inches=0.05)
        plt.clf()

