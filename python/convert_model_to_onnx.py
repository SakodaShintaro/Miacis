#!/usr/bin/env python3
import os
import argparse
import torch
from model.resnet import CategoricalNetwork
from constant import *
from model_dict import model_dict

parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=str)
parser.add_argument("--no_message", action="store_true")
args = parser.parse_args()

input_tensor = torch.randn([1, INPUT_CHANNEL_NUM, BOARD_SIZE, BOARD_SIZE]).cuda()


filename = os.path.splitext(os.path.basename(args.model_path))[0]
parts = filename.split("_")
block_num = None
channel_num = None

for part in parts:
    if "bl" in part:
        block_num = int(part.replace("bl", ""))
    if "ch" in part:
        channel_num = int(part.replace("ch", ""))

if not args.no_message:
    print(f"block_num = {block_num}, channel_num = {channel_num}")

model_name = parts[0]
model_class = model_dict[model_name]
model = CategoricalNetwork(INPUT_CHANNEL_NUM, block_num=block_num, channel_num=channel_num,
                           policy_channel_num=POLICY_CHANNEL_NUM, board_size=BOARD_SIZE)

saved_model = torch.load(args.model_path)
model.load_state_dict(saved_model)

model.eval()
model.cuda()

save_path = args.model_path.replace(".pt", ".onnx")
torch.onnx.export(model, input_tensor, save_path, dynamic_axes={"input": {0: "batch_size"}}, input_names=["input"])
if not args.no_message:
    print(f"export to {save_path}")
