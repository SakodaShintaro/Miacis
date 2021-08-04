#!/usr/bin/env python3
import time
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()

model = torch.jit.load(args.model_path)
model.cuda()

input_channel_num = 42
board_size = 9
policy_channel_num = 27
channel_num = 256
batch_size = 128

start = time.time()
num = 100

for _ in range(num):
    x = torch.randn((batch_size, input_channel_num, board_size, board_size)).cuda()
    p, v = model.forward(x)
    print(p.shape, v.shape)

elapsed = time.time() - start
ave = elapsed * 1000 / num
print(f"{ave:.4f} msec")
