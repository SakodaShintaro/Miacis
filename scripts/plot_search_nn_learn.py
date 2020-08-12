#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=(lambda x: x.split()),
                    default=["mcts_net", "proposed_model", "simple_mlp", "stacked_lstm"])
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

        for i in range(len(df.columns) - 3):
            loss = df[f"loss_{i + 1}"].to_numpy()
            if loss_name == "train":
                loss = transform(loss)
            plt.plot(step, loss)
            plt.text(step[-1], loss[-1], f"{i + 1}回探索後", color=plt.get_cmap("tab10")(i))
        plt.xlim((plt.xlim()[0], plt.xlim()[1] * 1.15))
        plt.xlabel("学習ステップ数")
        plt.ylabel("Policy損失")
        plt.savefig(f"{prefix}_{loss_name}.png", bbox_inches="tight", pad_inches=0.05)
        plt.clf()

# 全体をまとめてプロット
for loss_name in ["train", "valid"]:
    for i, prefix in enumerate(args.method):
        df = data_dict[prefix][loss_name]
        step = df["step"].to_numpy()
        if loss_name == "train":
            step = transform(step)
        last_index = len(df.columns) - 4
        loss = df[f"loss_{last_index + 1}"].to_numpy()
        if loss_name == "train":
            loss = transform(loss)
        plt.plot(step, loss)
        plt.text(step[-1], loss[-1], f"{prefix}({last_index + 1}回探索)", color=plt.get_cmap("tab10")(i))
    plt.xlim((plt.xlim()[0], plt.xlim()[1] * 1.5))
    plt.xlabel("学習ステップ数")
    plt.ylabel("Policy損失")
    plt.savefig(f"all_{loss_name}.png", bbox_inches="tight", pad_inches=0.05)
    plt.clf()
