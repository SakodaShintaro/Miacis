#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os
import japanize_matplotlib
import argparse

# ディレクトリの名前をコマンドライン引数として受け取る
parser = argparse.ArgumentParser()
parser.add_argument("--dirs", type=(lambda x: x.split()), required=True)
parser.add_argument("--labels", type=(lambda x: x.split()), required=True)
parser.add_argument("--prefix", type=str, default="supervised")
args = parser.parse_args()
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
            except BaseException:
                e = line[i].split(":")
                hour = float(e[0]) + float(e[1]) / 60 + float(e[2]) / 3600
                data[i].append(hour)
    return labels, data


TIME = 0
STEP = 1
POLICY_LOSS = 2
VALUE_LOSS = 3

train_labels = None
train_data = list()
valid_labels = None
valid_data = list()

for dir_name in args.dirs:
    if dir_name[-1] != "/":
        dir_name += "/"
    train_labels, t_data = get_labels_and_data(dir_name + f"{args.prefix}_train_log.txt")
    # trainデータは1ステップごとに記録されていて多すぎるのでSKIP個になるようにまとめて平均を取る
    SKIP = 200
    for i in range(len(t_data)):
        t_data[i] = t_data[i][0:len(t_data[i]) // SKIP * SKIP]
        t_data[i] = np.array(t_data[i]).reshape(SKIP, -1).mean(axis=1)
    train_data.append(t_data)
    valid_labels, v_data = get_labels_and_data(dir_name + f"{args.prefix}_valid_log.txt")
    valid_data.append(v_data)

# policy, valueそれぞれプロット
for x in [STEP]:
    for y in [POLICY_LOSS, VALUE_LOSS]:
        # train
        for name, data in zip(args.labels, train_data):
            plt.plot(data[x], data[y], label=name)
        plt.xlabel(train_labels[x])
        plt.ylabel(train_labels[y])
        if len(args.labels) > 1:
            plt.legend()
        plt.savefig("compare_train_" + train_labels[y] + ".png", bbox_inches="tight", pad_inches=0.1)
        plt.clf()

        # valid
        for name, data in zip(args.labels, valid_data):
            plt.plot(data[x], data[y], label=name, marker=".")
        plt.xlabel(valid_labels[x])
        plt.ylabel(valid_labels[y])
        if len(args.labels) > 1:
            plt.legend()
        plt.savefig("compare_valid_" + valid_labels[y] + ".png", bbox_inches="tight", pad_inches=0.1)
        plt.clf()

        # train and valid
        for name, data in zip(args.labels, train_data):
            plt.plot(data[x], data[y], label="train_" + name, linestyle="dashed")
        for name, data in zip(args.labels, valid_data):
            plt.plot(data[x], data[y], label="valid_" + name, marker=".")
        plt.xlabel(train_labels[x])
        plt.ylabel(train_labels[y])
        plt.legend()
        plt.savefig("compare_train_and_valid_" + train_labels[y] + ".png", bbox_inches="tight", pad_inches=0.1)
        plt.clf()
