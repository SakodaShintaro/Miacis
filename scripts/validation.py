#! /usr/bin/env python3
import os
import glob
from natsort import natsorted
import argparse
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--kifu_path", type=str, default="/root/data/floodgate_kifu/valid")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--init_model_step", type=int, default=0)
args = parser.parse_args()

# カレントディレクトリ内にある{prefix}_{step}.modelを評価する
curr_path = os.getcwd()
# ディレクトリ名が"/"で終わっていることの確認
if curr_path[-1] != "/":
    curr_path += "/"

# 結果を書き込むファイルを取得
f = open(curr_path + "validation_loss.txt", "a")

# ディレクトリにある以下のprefixを持ったパラメータを用いて検証損失の計算を行う
model_names = natsorted(glob.glob(curr_path + "*0.model"))

# このスクリプトがある場所
script_dir = os.path.dirname(os.path.abspath(__file__))

first = True

start = time.time()

# 各ステップの検証損失を計測
for model_name in model_names:
    # 最後に出てくるアンダーバーから.modelの直前までにステップ数が記録されているという前提
    step = int(model_name[model_name.rfind("_") + 1:model_name.find(".model")])

    # args.init_model_stepより小さいものは調べない
    if step < args.init_model_step:
        continue

    scalar_or_categorical = "scalar" if "sca" in model_name else "categorical"
    miacis_path = f"{script_dir}/../src/cmake-build-release/Miacis_shogi_{scalar_or_categorical}"
    command = f"checkVal\n{args.kifu_path}\n{args.batch_size}\n{model_name}\nquit"
    result = subprocess.run(f"echo '{command}' | {miacis_path}", shell=True, stdout=subprocess.PIPE)

    result_str = result.stdout.decode("utf8")
    result_sentences = result_str.split("\n")

    if first:
        print(result_sentences[1])
        f.write(result_sentences[1] + "\n")
        f.write("time\tstep\tpolicy_loss\tvalue_loss\n")
        first = False

    elapsed_time = int(time.time() - start)
    time_str = f"{elapsed_time // 3600:02d}:{(elapsed_time % 3600) // 60:02d}:{elapsed_time % 60:02d}"

    policy_loss, value_loss = result_sentences[2].split()

    # ファイルに書き込み
    tsv_str = f"{time_str}\t{step}\t{policy_loss}\t{value_loss}"
    print(tsv_str)
    f.write(f"{tsv_str}\n")
    f.flush()
