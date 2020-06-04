#!/usr/bin/env python3
import matplotlib.pyplot as plt
import japanize_matplotlib
import subprocess
f = open("search_log.txt")
lines = f.readlines()
index = 0

turn = 0

while index < len(lines):
    line = lines[index].strip()
    index += 1
    if line == "startSearch":
        # ターンが始まる
        pos_str = lines[index].strip()
        index += 1

        x = list()
        y = list()
        print(f"pos_str = {pos_str}")
        while True:
            string = lines[index].strip()
            index += 1
            if string == "endSearch":
                turn += 1
                break

            elements = string.split()
            if elements[1] != "string":
                continue
            value, prob = elements[2][1:12].split(':')
            x.append(float(value))
            y.append(float(prob))
        if turn % 2 == 0:
            x = x[::-1]
        plt.bar(x, y, width=2 / len(x))
        plt.xlabel("Value")
        plt.ylabel("Probability")
        plt.ylim((0, 100))
        plt.savefig(f"fig/{turn}.png", bbox_inches="tight", pad_inches=0.05)
        plt.cla()

subprocess.call('convert -delay 50 -loop 1 $(ls -v fig/*.png) fig/out.gif', shell=True)
