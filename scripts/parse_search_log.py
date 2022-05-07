#!/usr/bin/env python3
import matplotlib.pyplot as plt
import japanize_matplotlib
import subprocess
import urllib.parse
import requests
import argparse
from PIL import Image, ImageDraw

# ディレクトリの名前をコマンドライン引数として受け取る
parser = argparse.ArgumentParser()
parser.add_argument("--get_board", action="store_true")
parser.add_argument("--mix", action="store_true")
args = parser.parse_args()


f = open("search_log.txt")
lines = f.readlines()
index = 0

turn = 1

prefix = "http://sfenreader.appspot.com/sfen?sfen="

while index < len(lines):
    line = lines[index].strip()
    index += 1
    if line == "startSearch":
        # ターンが始まる
        pos_str = lines[index].strip("\n")
        index += 1

        x = list()
        y = list()
        while True:
            string = lines[index].strip()
            index += 1
            print(string)

            elements = string.split()
            if elements[1] == "nps":
                break
            l = elements[2][1:12].split(':')
            if len(l) < 2:
                continue
            value, prob = l
            x.append(float(value))
            y.append(float(prob))
        if turn % 2 == 0:
            x = x[::-1]
        plt.bar(x, y, width=2 / len(x))
        plt.xlabel("Value(探索前)")
        plt.ylabel("Probability")
        plt.ylim((0, 100))
        plt.text(0.2, 95, f"turn = {turn}")
        plt.savefig(f"value1_{turn}.png", bbox_inches="tight", pad_inches=0.05)
        plt.cla()

        x.clear()
        y.clear()

        while True:
            string = lines[index].strip()
            index += 1
            print(string)

            elements = string.split()
            if elements[1] == "nps":
                break
            l = elements[2][1:12].split(':')
            if len(l) < 2:
                continue
            value, prob = l
            x.append(float(value))
            y.append(float(prob))
        if turn % 2 == 0:
            x = x[::-1]
        plt.bar(x, y, width=2 / len(x))
        plt.xlabel("Value(探索後)")
        plt.ylabel("Probability")
        plt.ylim((0, 100))
        plt.text(0.2, 95, f"turn = {turn}")
        plt.savefig(f"value2_{turn}.png", bbox_inches="tight", pad_inches=0.05)
        plt.cla()

        print(pos_str)
        if args.get_board:
            r = requests.get(prefix + urllib.parse.quote(pos_str))
            with open(f"board_{turn}.png", "wb") as out_file:
                out_file.write(r.content)

        turn += 1

# ValueをGIF化
# subprocess.call('convert -delay 25 -loop 1 $(ls -v value_*.png) value_out.gif', shell=True)

for t in range(1, turn + 1):
    subprocess.call(f'convert +append value1_{t}.png value2_{t}.png both_value{t}.png', shell=True)

# 画像の結合
if args.mix:
    for t in range(1, turn + 1):
        subprocess.call(f'convert +append board_{t}.png both_value{t}.png mixed_{t}.png', shell=True)

    # GIF化
    subprocess.call('convert -delay 25 -loop 1 $(ls -v mixed_*.png) mixed_out.gif', shell=True)
