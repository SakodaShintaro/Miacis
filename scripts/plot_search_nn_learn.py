#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-prefix", type=str, required=True)
args = parser.parse_args()

M = 10
PLOT_NUM = 20


def transform(arr):
    arr = arr[0:len(arr) // PLOT_NUM * PLOT_NUM]
    arr = np.array(arr)
    arr = arr.reshape([PLOT_NUM, -1])
    return arr.mean(1)


for loss_name in ["train", "valid"]:
    f = open(f"./{args.prefix}_{loss_name}_log.txt")

    step = list()
    losses = None

    # データ読み込み
    for line in f:
        elements = line.strip().split()
        if elements[0] == "time":
            # ラベルの量によってMなどを調整
            M = len(elements) - 3
            losses = [list() for _ in range(M)]
            continue
        step.append(int(elements[2]))
        for i in range(M):
            losses[i].append(float(elements[3 + i]))

    if loss_name == "train":
        # 数が多いので適当な間隔で平均を取る
        step = transform(step)

        for i in range(M):
            losses[i] = transform(losses[i])

    for i in range(M):
        plt.plot(step, losses[i])
        plt.text(step[-1], losses[i][-1], f"{i + 1}回探索後", color=plt.get_cmap("tab10")(i))
    plt.xlim((plt.xlim()[0], plt.xlim()[1] * 1.15))
    plt.xlabel("学習ステップ数")
    plt.ylabel("Policy損失")
    plt.savefig(f"{args.prefix}_{loss_name}.png", bbox_inches="tight", pad_inches=0.05)
    plt.clf()
