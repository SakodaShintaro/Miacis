#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=(lambda x: x.split()),
                    default=["mcts_net", "proposed_model", "stacked_lstm"])
args = parser.parse_args()

PLOT_NUM = 20


def transform(arr):
    arr = arr[0:len(arr) // PLOT_NUM * PLOT_NUM]
    arr = np.array(arr)
    arr = arr.reshape([PLOT_NUM, -1])
    return arr.mean(1)


data_dict = dict()

# まずデータを取得しつつ各損失単体をプロット
for prefix in args.method:
    data_dict[prefix] = dict()
    for loss_name in ["train", "valid"]:
        df = pd.read_csv(f"./{prefix}_{loss_name}_log.txt", delimiter="\t")

        data_dict[prefix][loss_name] = df

        step = df["step"].to_numpy()
        if loss_name == "train":
            step = transform(step)

        loss_num = len(df.columns) - 3
        for i in range(loss_num):
            loss = df[f"loss_{i}"].to_numpy()
            if loss_name == "train":
                loss = transform(loss)
            color = [0.0, 0.5, i / loss_num]
            plt.plot(step, loss, color=color, label=f"{i}回探索後" if loss_name == "valid" else (
                f"{loss_num - 1}回探索後の損失" if i == 0 else f"{i}回目探索の寄与と確率の積"),
                     linestyle=("dashed" if i == 0 else "solid"))
            # plt.text(step[-1], loss[-1], f"{i + 1}回探索後", color=color)
        plt.legend()
        # plt.xlim((plt.xlim()[0], plt.xlim()[1] * 1.15))
        plt.xlabel("学習ステップ数")
        plt.ylabel("Policy損失")
        plt.savefig(f"{prefix}_{loss_name}.png", bbox_inches="tight", pad_inches=0.05)
        plt.clf()

        # 最終損失だけをプロット
        x = [i for i in range(loss_num)]
        y = [df[f"loss_{i}"].to_numpy()[-1] for i in range(loss_num)]
        plt.plot(x, y, marker=".")
        plt.xlabel("探索回数")
        plt.ylabel("Policy損失")
        plt.savefig(f"{prefix}_{loss_name}_final.png", bbox_inches="tight", pad_inches=0.05)
        plt.clf()

# 全体をまとめてプロット
for loss_name in ["train", "valid"]:
    for i, prefix in enumerate(args.method):
        df = data_dict[prefix][loss_name]
        step = df["step"].to_numpy()
        if loss_name == "train":
            step = transform(step)
        last_index = len(df.columns) - 4
        loss = df[f"loss_{last_index}"].to_numpy()
        if loss_name == "train":
            loss = transform(loss)
        plt.plot(step, loss)
        plt.text(step[-1], loss[-1], f"{prefix}({last_index}回探索)", color=plt.get_cmap("tab10")(i))
    plt.xlim((plt.xlim()[0], plt.xlim()[1] * 1.5))
    plt.xlabel("学習ステップ数")
    plt.ylabel("Policy損失")
    plt.savefig(f"all_{loss_name}.png", bbox_inches="tight", pad_inches=0.05)
    plt.clf()


def time_str2time_int(time_str):
    h, m, s = time_str.split(':')
    return int(h) + int(m) / 60 + int(s) / 3600


for loss_name in ["train", "valid"]:
    for i, prefix in enumerate(args.method):
        df = data_dict[prefix][loss_name]
        time = df["time"].to_numpy()
        time = [time_str2time_int(v) for v in time]
        if loss_name == "train":
            time = transform(time)
        last_index = len(df.columns) - 4
        loss = df[f"loss_{last_index}"].to_numpy()
        if loss_name == "train":
            loss = transform(loss)
        plt.plot(time, loss, label=f"{prefix}({last_index}回探索)")
        # plt.text(time[-1], loss[-1], f"{prefix}({last_index + 1}回探索)", color=plt.get_cmap("tab10")(i))
    # plt.xlim((plt.xlim()[0], plt.xlim()[1] * 1.5))
    plt.legend()
    plt.xlabel("学習時間(h)")
    plt.ylabel("Policy損失")
    plt.savefig(f"all_time_{loss_name}.png", bbox_inches="tight", pad_inches=0.05)
    plt.clf()
