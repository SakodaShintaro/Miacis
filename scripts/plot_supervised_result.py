#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os
import japanize_matplotlib
import argparse

# ディレクトリの名前をコマンドライン引数として受け取る
parser = argparse.ArgumentParser()
parser.add_argument("-dirs", type=(lambda x: x.split()))
parser.add_argument("--labels", type=(lambda x: x.split()), default=None)
args = parser.parse_args()
if args.labels is None:
    args.labels = [""]

assert len(args.dirs) == len(args.labels)


def get_labels_and_data(file_name):
    f = open(file_name)
    labels = [_ for _ in f.readline().strip().split("\t")]
    data = [list() for _ in range(len(labels))]
    for line in f:
        line = line.strip()
        line = line.split("\t")
        for i in range(len(line)):
            try:
                data[i].append(float(line[i]))
            except:
                e = line[i].split(":")
                hour = float(e[0]) + float(e[1]) / 60 + float(e[2]) / 3600
                data[i].append(hour)
    return labels, data


TIME = 0
EPOCH = 1
STEP = 2
POLICY_LOSS = 3
VALUE_LOSS = 4
ELO_RATE = 5

train_labels = None
train_data = list()
valid_labels = None
valid_data = list()
battle_result = list()

for dir_name in args.dirs:
    if dir_name[-1] != "/":
        dir_name += "/"
    train_labels, t_data = get_labels_and_data(dir_name + "supervised_train_log.txt")
    # trainデータは1ステップごとに記録されていて多すぎるのでSKIP個になるようにまとめて平均を取る
    SKIP = 200
    for i in range(len(t_data)):
        t_data[i] = np.array(t_data[i]).reshape(SKIP, -1).mean(axis=1)
    train_data.append(t_data)
    valid_labels, v_data = get_labels_and_data(dir_name + "supervised_valid_log.txt")
    valid_data.append(v_data)

    # 対局結果を取得
    # 対局結果はresult.txtにある
    result_file_name = dir_name + "/result.txt"
    if not os.path.exists(result_file_name):
        print("result.txt does not exist in ", dir_name)
        continue

    steps = list()
    rates = list()
    for line in open(result_file_name):
        # 空白区切りで"相対レート"という要素の次にレートが記録されていることを前提とする
        elements = line.strip().split()
        for e in elements:
            if "ステップ" in e:
                steps.append(int(e.replace("ステップ", "")))
        if "相対レート" in elements:
            rates.append(float(elements[elements.index("相対レート") + 1]))

    c = zip(steps, rates)
    c = sorted(c)
    steps, rates = zip(*c)

    battle_result.append((steps, rates))

# policy, valueそれぞれプロット
for x in [STEP]:
    for y in [POLICY_LOSS, VALUE_LOSS]:
        # train
        for name, data in zip(args.labels, train_data):
            plt.plot(data[x], data[y], label=name)
        plt.xlabel(train_labels[x])
        plt.ylabel(train_labels[y])
        plt.legend()
        plt.savefig("compare_train_" + train_labels[y] + ".png", bbox_inches="tight", pad_inches=0.1)
        plt.clf()

        # valid
        for name, data in zip(args.labels, valid_data):
            plt.plot(data[x], data[y], label=name)
        plt.xlabel(valid_labels[x])
        plt.ylabel(valid_labels[y])
        plt.legend()
        plt.savefig("compare_valid_" + valid_labels[y] + ".png", bbox_inches="tight", pad_inches=0.1)
        plt.clf()

        # train and valid
        for name, data in zip(args.labels, train_data):
            plt.plot(data[x], data[y], label="train_" + name, linestyle="dashed")
        for name, data in zip(args.labels, valid_data):
            plt.plot(data[x], data[y], label="valid_" + name)
        plt.xlabel(train_labels[x])
        plt.ylabel(train_labels[y])
        plt.legend()
        plt.savefig("compare_train_and_valid_" + train_labels[y] + ".png", bbox_inches="tight", pad_inches=0.1)
        plt.clf()

# 対局結果をプロット
for name, data in zip(args.labels, battle_result):
    plt.plot(data[0], data[1], label=name)
plt.legend()
plt.savefig("compare_battle_result.png", bbox_inches="tight", pad_inches=0.1)
plt.clf()
