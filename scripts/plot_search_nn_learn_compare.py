#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
import argparse
import pandas as pd
import os
import glob

# markers = [".", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "*", "h", "H", "+", "x", "D"]
markers = ["None", ".", "v", "s", "x"]

parser = argparse.ArgumentParser()
parser.add_argument("-dirs", type=(lambda x: x.split()))
parser.add_argument("--labels", type=(lambda x: x.split()), default=None)
parser.add_argument("--max_length", type=int, default=24)
parser.add_argument("--fix", action="store_true")
args = parser.parse_args()
if args.labels is None:
    args.labels = [""]

PLOT_NUM = 20
loss_num = 11


def transform(arr):
    arr = arr[0:len(arr) // PLOT_NUM * PLOT_NUM]
    arr = np.array(arr)
    arr = arr.reshape([PLOT_NUM, -1])
    return arr.mean(1)


data_dict = dict()

suffixes = ["_train_log.txt", "_valid_log.txt"]


def time_str2time_int(time_str):
    h, m, s = time_str.split(':')
    return int(h) + int(m) / 60 + int(s) / 3600


for dir, label in zip(args.dirs, args.labels):
    data_dict[label] = dict()
    for suffix in suffixes:
        files = glob.glob(f"{dir}/*{suffix}")
        path = files[0]
        print(path)
        if os.path.exists(path):
            df = pd.read_csv(path, delimiter="\t")
            data_dict[label][suffix] = df

# 全体をまとめてプロット
for suffix in suffixes:
    # base_policyを取得
    base_policy_loss = data_dict[args.labels[0]][suffix]
    df = data_dict[args.labels[0]][suffix]
    step = df["step"].to_numpy()
    # step = df["time"].to_numpy()
    loss = df["base_policy"].to_numpy()
    plt.plot(step, loss, linestyle="dashed", label="Policyネットワーク")
    # plt.text(step[-1], loss[-1], f"Policyネットワーク", color=plt.get_cmap("tab10")(len(args.labels)))

    for i, label in enumerate(args.labels):
        df = data_dict[label][suffix]
        step = df["step"].to_numpy()
        # step = df["time"].to_numpy()
        # step = [time_str2time_int(v) for v in step]
        loss = df[f"last_policy_loss"].to_numpy()

        if suffix == suffixes[0]:
            step = transform(step)
            loss = transform(loss)

        plt.plot(step, loss, label=f"{label}", marker=markers[i + 1])
        # plt.text(step[-1], loss[-1], f"{label}", color=plt.get_cmap("tab10")(i))

    plt.legend()
    # plt.xlim((plt.xlim()[0], plt.xlim()[1] * 1.4))
    plt.xlabel("学習ステップ数")
    plt.ylabel("Policy損失")
    plt.savefig(f"all_policy.png", bbox_inches="tight", pad_inches=0.05)
    plt.clf()


# 探索結果の方を描画
# まずデータを取得
i = 0
for dir, label in zip(args.dirs, args.labels):
    files = glob.glob(f"{dir}/valid_with_search.txt")

    if len(files) == 0:
        continue

    path = files[0]
    df = pd.read_csv(path, delimiter=",")

    step = df["探索回数"].to_numpy()
    loss = df[f"policy_loss"].to_numpy()

    step = step[1:]
    loss = loss[1:]

    if label == args.labels[0]:
        x = [i for i in range(1, args.max_length + 1)]
        base_loss = base_policy_loss["base_policy"].to_numpy()
        plt.plot(x, [base_loss[0] for _ in range(args.max_length)], label="Policyネットワーク", linestyle="dashed",
                 color=plt.get_cmap("tab10")(0))

    i += 1
    plt.plot(step, loss, label=label, marker=markers[i], color=plt.get_cmap("tab10")(i))

plt.legend()
plt.xlabel("探索回数")
plt.ylabel("Policy損失")
if args.fix:
    plt.ylim((0.6, 0.925))
plt.savefig(f"all_policy_with_search.png", bbox_inches="tight", pad_inches=0.05)
