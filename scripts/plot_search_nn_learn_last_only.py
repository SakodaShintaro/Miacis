#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
import pandas as pd
import glob

PLOT_NUM = 20


def transform(arr):
    arr = arr[0:len(arr) // PLOT_NUM * PLOT_NUM]
    arr = np.array(arr)
    arr = arr.reshape([PLOT_NUM, -1])
    return arr.mean(1)


suffix = "_train_log.txt"
files = glob.glob("*" + suffix)
prefix = files[0].replace(suffix, "")
loss_num = (1 if prefix == "simple_mlp" else 11)

# まずデータを取得しつつ各損失単体をプロット
for pol_or_val in ["policy", "value"]:
    for train_or_valid in ["train", "valid"]:
        df = pd.read_csv(f"./{prefix}_{train_or_valid}_log.txt", delimiter="\t")

        step = df["step"].to_numpy()
        if train_or_valid == "train":
            step = transform(step)

        loss = df[f"last_{pol_or_val}_loss"].to_numpy()
        if train_or_valid == "train":
            loss = transform(loss)
        # plt.plot(step, loss, color=color, label=f"{i}回探索後" if loss_name == "valid" else (
        #     f"{loss_num - 1}回探索後の損失" if i == 0 else f"{i}回目探索の寄与と確率の積"),
        #          linestyle=("dashed" if i == 0 else "solid"))
        plt.plot(step, loss, label="Transformer(入力長=10)")

        base_loss = df[f"base_{pol_or_val}"].to_numpy()
        if train_or_valid == "train":
            base_loss = transform(base_loss)
        plt.plot(step, base_loss, label="探索なし")

        # plt.text(step[-1], loss[-1], f"{i + 1}回探索後", color=color)
        plt.legend()
        # plt.xlim((plt.xlim()[0], plt.xlim()[1] * 1.15))
        plt.xlabel("学習ステップ数")
        plt.ylabel(f"{pol_or_val}損失")
        plt.savefig(f"{prefix}_{train_or_valid}_{pol_or_val}.png", bbox_inches="tight", pad_inches=0.05)
        plt.clf()

        # # 最終損失だけをプロット
        # x = [i for i in range(loss_num)]
        # y = [df[f"{pol_or_val}_loss_{i}"].to_numpy()[-1] for i in range(loss_num)]
        # plt.plot(x, y, marker=".")
        # plt.xlabel("探索回数")
        # plt.ylabel(f"{pol_or_val}損失")
        # plt.savefig(f"{prefix}_{train_or_valid}_{pol_or_val}_final.png", bbox_inches="tight", pad_inches=0.05)
        # plt.clf()
