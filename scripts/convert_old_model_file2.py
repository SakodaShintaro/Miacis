#!/usr/bin/env python3
import glob
import os
import re
from natsort import natsorted
from generate_cnn_model import *

def load_conv_and_norm(dst, src):
    dst.conv_.weight.data = src.conv_.weight.data
    dst.norm_.weight.data = src.norm_.weight.data
    dst.norm_.bias.data = src.norm_.bias.data
    dst.norm_.running_mean = src.norm_.running_mean
    dst.norm_.running_var = src.norm_.running_var

source_model_name = "/root/Miacis/build/shogi_cat_bl20_ch512.model"

# 1番目のモデル名からブロック数,チャンネル数を読み取る.これらは1ディレクトリ内で共通だという前提
basename_without_ext = os.path.splitext(os.path.basename(source_model_name))[0]
parts = basename_without_ext.split("_")
block_num = None
channel_num = None
for p in parts:
    if "bl" in p:
        block_num = int(re.sub("\\D", "", p))
    elif "ch" in p:
        channel_num = int(re.sub("\\D", "", p))

# インスタンス生成
input_channel_num = 42
board_size = 9
policy_channel_num = 27
model = CategoricalNetwork(input_channel_num, block_num, channel_num, policy_channel_num, board_size)

# 各モデルファイルのパラメータをコピーしてTorchScriptとして保存
source = torch.jit.load(source_model_name).cpu()

# first_conv
load_conv_and_norm(model.encoder_.first_conv_and_norm_, source.encoder_.first_conv_and_norm_)

print(source.encoder_.__dict__["_modules"]["blocks"].__dict__.keys())

# block
for i, v in enumerate(model.encoder_.__dict__["_modules"]["blocks"]):
    source_m = source.encoder_.__dict__["_modules"]["blocks"].__dict__["_modules"][f"block{i}"]
    load_conv_and_norm(v.conv_and_norm0_, source_m.conv_and_norm0_)
    load_conv_and_norm(v.conv_and_norm1_, source_m.conv_and_norm1_)
    v.linear0_.weight.data = source_m.linear0_.weight.data
    v.linear1_.weight.data = source_m.linear1_.weight.data

# policy_conv
model.policy_head_.policy_conv_.weight.data = source.policy_head_.policy_conv_.weight.data
model.policy_head_.policy_conv_.bias.data = source.policy_head_.policy_conv_.bias.data

# value_conv_norm_
load_conv_and_norm(model.value_head_.value_conv_and_norm_, source.value_head_.value_conv_and_norm_)

# value_linear
model.value_head_.value_linear0_.weight.data = source.value_head_.value_linear0_.weight.data
model.value_head_.value_linear0_.bias.data = source.value_head_.value_linear0_.bias.data
model.value_head_.value_linear1_.weight.data = source.value_head_.value_linear1_.weight.data
model.value_head_.value_linear1_.bias.data = source.value_head_.value_linear1_.bias.data

input_data = torch.ones([1, input_channel_num, board_size, board_size])
model.eval()
script_model = torch.jit.trace(model, input_data)
# script_model = torch.jit.script(model)
model_path = f"here.model"
script_model.save(model_path)
print(f"{model_path}にパラメータを保存")
