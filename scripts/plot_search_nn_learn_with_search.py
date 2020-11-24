#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import japanize_matplotlib
import argparse
import glob
import os
from natsort import natsorted
import subprocess

# カレントディレクトリにあるパラメータを使って,探索数nの探索を行って,それをファイルに書き出して,それを読み出してプロットする

# 引数の設定
parser = argparse.ArgumentParser()
parser.add_argument("--search_limit", type=int, default=24)
parser.add_argument("--rate_threshold", type=int, default=2290)
parser.add_argument("--print_start", type=int, default=1)
args = parser.parse_args()

# カレントディレクトリ内にある{prefix}_{step}.modelを評価する
curr_path = os.getcwd()
# ディレクトリ名が"/"で終わっていることの確認
if curr_path[-1] != "/":
    curr_path += "/"

# ディレクトリにある以下のprefixを持ったパラメータを取り出す
model_names = natsorted(glob.glob(curr_path + "*0.model"))

# 最終ステップのものを使う
model_name = model_names[-1]

# このスクリプト自体があるパスを取得
script_dir = os.path.dirname(os.path.abspath(__file__))

proc = subprocess.Popen(
    [f"{script_dir}/../src/cmake-build-release/Miacis_othello_scalar"],
    cwd=f"{script_dir}/../src/cmake-build-release",
    encoding="UTF-8",
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE)
# 最初はCUDAの利用判定が一行入るので読み飛ばす
proc.stdout.readline()

command_name = model_name.split("/")
command_name = command_name[-1]
command_name = command_name.split("_")
command_name = command_name[0:-1]
command_name = [word.capitalize() for word in command_name]
command_name = [("LSTM" if word == "Lstm" else word) for word in command_name]
command_name = [("MCTS" if word == "Mcts" else word) for word in command_name]
command_name = "".join(command_name)
command_name = "valid" + command_name

message = command_name
proc.stdin.write(message + "\n")
proc.stdin.flush()

# 使用棋譜の指定
proc.stdin.write("/home/sakoda/othello_valid_kifu\n")

# レートの閾値設定
proc.stdin.write(f"{args.rate_threshold}\n")

# バッチサイズ設定
proc.stdin.write("256\n")

# 探索回数設定
proc.stdin.write(f"{args.search_limit}\n")

# モデル名
proc.stdin.write(f"{model_name}\n")

proc.stdin.flush()

while True:
    line = proc.stdout.readline()
    elements = line.split()
    print(line, end="")
    if len(elements) == (args.search_limit + 1) * 2 + 3:
        break

x = [i for i in range(args.search_limit + 1)]
policy_loss = [float(elements[i * 2]) for i in range(args.search_limit + 1)]
value_loss = [float(elements[i * 2 + 1]) for i in range(args.search_limit + 1)]

base_policy_loss = float(elements[-3])
base_value_loss = float(elements[-2])

with open("valid_with_search.txt", "w") as f:
    print("探索回数,policy_loss,value_loss")
    for i in range(args.search_limit + 1):
        print(x[i], policy_loss[i], value_loss[i])
        f.write(f"{x[i]},{policy_loss[i]},{value_loss[i]}\n")

x = x[args.print_start:]
policy_loss = policy_loss[args.print_start:]
value_loss = value_loss[args.print_start:]

plt.plot(x, policy_loss, marker=".", label="Proposed Model")
plt.plot(x, [base_policy_loss for _ in range(len(x))], label="base_policy", linestyle="dashed")

plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))

plt.xlabel("入力系列長")
plt.ylabel("Policy損失")
plt.legend()
plt.savefig("search_num-policy_loss.png", bbox_inches="tight", pad_inches=0.05)
