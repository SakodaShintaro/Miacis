#!/usr/bin/env python3
import glob
import os
import re
from natsort import natsorted
from generate_torch_script_model import *


# batch_normがある場合はちょっと特殊なので関数として切り出しておく
def load_conv_and_norm(dst, src):
    dst.conv_.weight.data = src.conv_.weight.data
    dst.norm_.weight.data = src.norm_.weight.data
    dst.norm_.bias.data = src.norm_.bias.data
    dst.norm_.running_mean = src.norm_.running_mean
    dst.norm_.running_var = src.norm_.running_var


parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, required=True)
parser.add_argument("--game", default="shogi", choices=["shogi", "othello"])
args = parser.parse_args()

if args.game == "shogi":
    input_channel_num = 42
    board_size = 9
    policy_channel_num = 27
elif args.game == "othello":
    input_channel_num = 2
    board_size = 8
    policy_channel_num = 2

# ディレクトリにある以下のprefixを持ったパラメータを用いて対局を行う
source_model_names = natsorted(glob.glob(f"{args.source_dir}/*.model"))

# 1番目のモデル名からブロック数,チャンネル数を読み取る.これらは1ディレクトリ内で共通だという前提
basename_without_ext = os.path.splitext(os.path.basename(source_model_names[0]))[0]
parts = basename_without_ext.split("_")
block_num = None
channel_num = None
for p in parts:
    if "bl" in p:
        block_num = int(re.sub("\\D", "", p))
    elif "ch" in p:
        channel_num = int(re.sub("\\D", "", p))

# インスタンス生成
model = CategoricalNetwork(input_channel_num, block_num, channel_num, policy_channel_num, board_size)

# 各モデルファイルのパラメータをコピーしてTorchScriptとして保存
for source_model_name in source_model_names:
    source = torch.jit.load(source_model_name).cpu()

    # first_conv
    load_conv_and_norm(model.encoder_.first_conv_and_norm_, source.state_first_conv_and_norm_)

    # block
    for i, v in enumerate(model.encoder_.__dict__["_modules"]["blocks"]):
        source_m = source.__dict__["_modules"][f"state_blocks_{i}"]
        load_conv_and_norm(v.conv_and_norm0_, source_m.conv_and_norm0_)
        load_conv_and_norm(v.conv_and_norm1_, source_m.conv_and_norm1_)
        v.linear0_.weight.data = source_m.linear0_.weight.data
        v.linear1_.weight.data = source_m.linear1_.weight.data

    # policy_conv
    model.policy_head_.policy_conv_.weight.data = source.policy_conv_.weight.data
    model.policy_head_.policy_conv_.bias.data = source.policy_conv_.bias.data

    # value_conv_norm_
    load_conv_and_norm(model.value_head_.value_conv_and_norm_, source.value_conv_and_norm_)

    # value_linear
    model.value_head_.value_linear0_.weight.data = source.value_linear0_.weight.data
    model.value_head_.value_linear0_.bias.data = source.value_linear0_.bias.data
    model.value_head_.value_linear1_.weight.data = source.value_linear1_.weight.data
    model.value_head_.value_linear1_.bias.data = source.value_linear1_.bias.data

    input_data = torch.ones([1, input_channel_num, board_size, board_size])
    model.eval()
    script_model = torch.jit.trace(model, input_data)
    # script_model = torch.jit.script(model)
    model_path = f"{args.game}_{os.path.basename(source_model_name)}"
    script_model.save(model_path)
    print(f"{model_path}にパラメータを保存")
