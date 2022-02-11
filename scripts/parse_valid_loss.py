#!/usr/bin/env python3

import argparse
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import japanize_matplotlib

parser = argparse.ArgumentParser()
parser.add_argument("tsv_file1", type=str)
parser.add_argument("tsv_file2", type=str)
args = parser.parse_args()

label1 = "ResNet"
label2 = "ViT"

df1 = pd.read_csv(args.tsv_file1, sep="\t", header=None)
df2 = pd.read_csv(args.tsv_file2, sep="\t", header=None)

#########
# 4分類 #
#########
# grid_result = defaultdict(int)

# for i in tqdm(range(len(df1))):
#     sfen1, move_str1, move_prob1, value_label1, policy_loss1, value_loss1 = df1.iloc[i]
#     sfen2, move_str2, move_prob2, value_label2, policy_loss2, value_loss2 = df2.iloc[i]

#     assert sfen1 == sfen2
#     grid_result[(policy_loss1 > policy_loss2, value_loss1 > value_loss2)] += 1

# print(grid_result)


##########
# 手数毎 #
##########
# INTERVAL = 10

# turn_policy_loss1 = defaultdict(float)
# turn_policy_loss2 = defaultdict(float)
# turn_value_loss1 = defaultdict(float)
# turn_value_loss2 = defaultdict(float)
# turn_num = defaultdict(int)

# for i in tqdm(range(len(df1))):
#     sfen1, move_str1, move_prob1, value_label1, policy_loss1, value_loss1 = df1.iloc[i]
#     sfen2, move_str2, move_prob2, value_label2, policy_loss2, value_loss2 = df2.iloc[i]

#     turn_number = int(sfen1.split(" ")[-1])
#     turn_number = turn_number // INTERVAL * INTERVAL

#     turn_policy_loss1[turn_number] += policy_loss1
#     turn_policy_loss2[turn_number] += policy_loss2
#     turn_value_loss1[turn_number] += value_loss1
#     turn_value_loss2[turn_number] += value_loss2
#     turn_num[turn_number] += 1


# def sorted_dict(d):
#     return dict(sorted(d.items(), key=lambda x: x[0]))


# turn_policy_loss1 = sorted_dict(turn_policy_loss1)
# turn_policy_loss2 = sorted_dict(turn_policy_loss2)
# turn_value_loss1 = sorted_dict(turn_value_loss1)
# turn_value_loss2 = sorted_dict(turn_value_loss2)
# turn_num = sorted_dict(turn_num)

# for key, val in turn_num.items():
#     turn_policy_loss1[key] /= val
#     turn_policy_loss2[key] /= val
#     turn_value_loss1[key] /= val
#     turn_value_loss2[key] /= val

# print(turn_policy_loss1)
# print(turn_policy_loss2)

# plt.plot(turn_policy_loss1.keys(), turn_policy_loss1.values(), label=label1)
# plt.plot(turn_policy_loss2.keys(), turn_policy_loss2.values(), label=label2)
# plt.xlabel("手数")
# plt.ylabel("Policy損失")
# plt.legend()
# plt.savefig("turn_policy_loss.png", bbox_inches="tight", pad_inches=0.05)
# plt.cla()

# plt.plot(turn_value_loss1.keys(), turn_value_loss1.values(), label=label1)
# plt.plot(turn_value_loss2.keys(), turn_value_loss2.values(), label=label2)
# plt.xlabel("手数")
# plt.ylabel("Value損失")
# plt.legend()
# plt.savefig("turn_value_loss.png", bbox_inches="tight", pad_inches=0.05)
# plt.cla()


######################
# ソートして定性評価 #
######################
# df1 = df1.set_axis(["SFEN", "move_str", "move_prob", "value_label", "policy_loss", "value_loss"], axis=1)
# df2 = df2.set_axis(["SFEN", "move_str", "move_prob", "value_label", "policy_loss", "value_loss"], axis=1)
# df2 = df2.sort_values("value_loss", ascending=False)
# for i in range(len(df2[0:10])):
#     sfen2, move_str2, move_prob2, value_label2, policy_loss2, value_loss2 = df2.iloc[i]
#     print(sfen2, f"\t正解指し手:{move_str2}", move_prob2, value_label2, f"{policy_loss2:.3f}", f"{value_loss2:.4f}")

#     # 1のほうも見る
#     sfen1, move_str1, move_prob1, value_label1, policy_loss1, value_loss1 = df1[df1["SFEN"] == sfen2].iloc[0]
#     print(sfen1, f"\t正解指し手:{move_str1}", move_prob1, value_label1, f"{policy_loss1:.3f}", f"{value_loss1:.4f}")
#     print()


##############
# 相関を見る #
##############
policy_loss_list1 = list()
policy_loss_list2 = list()
value_loss_list1 = list()
value_loss_list2 = list()

for i in tqdm(range(len(df1))):
    sfen1, move_str1, move_prob1, value_label1, policy_loss1, value_loss1 = df1.iloc[i]
    sfen2, move_str2, move_prob2, value_label2, policy_loss2, value_loss2 = df2.iloc[i]

    policy_loss_list1.append(policy_loss1)
    policy_loss_list2.append(policy_loss2)

    value_loss_list1.append(value_loss1)
    value_loss_list2.append(value_loss2)

plt.scatter(policy_loss_list1, policy_loss_list2, s=1.25)
plt.xlabel(f"{label1}のpolicy損失")
plt.ylabel(f"{label2}のpolicy損失")
plt.savefig("scatter_policy_loss.png", bbox_inches="tight", pad_inches=0.05)
plt.cla()

plt.scatter(value_loss_list1, value_loss_list2, s=1.25)
plt.xlabel(f"{label1}のValue損失")
plt.ylabel(f"{label2}のValue損失")
plt.savefig("scatter_value_loss.png", bbox_inches="tight", pad_inches=0.05)
plt.cla()
