#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os
import japanize_matplotlib
import argparse

f = open("./mcts_net_train_log.txt")

step = list()
M = 10
INTERVAL = 250
losses = [list() for _ in range(M)]

for line in f:
    elements = line.strip().split()
    print(elements)
    if elements[0] == "time":
        continue
    step.append(int(elements[2]))
    for i in range(M):
        losses[i].append(float(elements[4 + i]))

def transform(arr):
    arr = np.array(arr)
    arr = arr.reshape([-1, INTERVAL])
    return arr.mean(1)

step = transform(step)

for i in range(M):
    losses[i] = transform(losses[i])

for i in range(M):
    plt.plot(step, losses[i])
    plt.text(step[-1], losses[i][-1], f"{i + 1}回探索後", color=plt.get_cmap("tab10")(i))
plt.xlim((plt.xlim()[0], plt.xlim()[1] * 1.15))
plt.xlabel("学習ステップ数")
plt.ylabel("Policy損失")
plt.savefig("mcts_net_learn.png", bbox_inches="tight", pad_inches=0.05)
