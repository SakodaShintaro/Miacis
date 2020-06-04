#!/usr/bin/env python3
import matplotlib.pyplot as plt
import japanize_matplotlib
import subprocess
import urllib.parse
import requests
import argparse

# ディレクトリの名前をコマンドライン引数として受け取る
parser = argparse.ArgumentParser()
parser.add_argument("--get_board", action="store_true")
parser.add_argument("--mix", action="store_true")
args = parser.parse_args()


f = open("search_log.txt")
lines = f.readlines()
index = 0

turn = 0

prefix = "http://sfenreader.appspot.com/sfen?sfen="

while index < len(lines):
    line = lines[index].strip()
    index += 1
    if line == "startSearch":
        # ターンが始まる
        pos_str = lines[index].strip()
        index += 1

        x = list()
        y = list()
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
        plt.text(0.2, 95, f"turn = {turn}")
        plt.savefig(f"value_{turn}.png", bbox_inches="tight", pad_inches=0.05)
        plt.cla()

        print(pos_str)
        if args.get_board:
            r = requests.get(prefix + urllib.parse.quote(pos_str))
            with open(f"board_{turn}.png", "wb") as out_file:
                out_file.write(r.content)

# ValueをGIF化
subprocess.call('convert -delay 25 -loop 1 $(ls -v value_*.png) value_out.gif', shell=True)

# 画像の結合
if args.mix:
    for t in range(1, turn + 1):
        subprocess.call(f'convert +append board_{t}.png value_{t}.png mixed_{t}.png', shell=True)

    # GIF化
    subprocess.call('convert -delay 25 -loop 1 $(ls -v mixed_*.png) mixed_out.gif', shell=True)
