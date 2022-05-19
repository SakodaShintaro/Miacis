#!/usr/bin/env python3
import torch
from constant import *
from dataset_random_move import RandomMoveDataSet
import argparse
from common import seconds_to_pretty_str
from scheduler import LinearWarmupAndCooldownScheduler
import time
from model_dict import model_dict

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--weight_decay", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--max_step", type=int, default=60000)
parser.add_argument("--learn_rate_decay_mode", type=int, default=0)
parser.add_argument("--learn_rate_decay_period", type=int, default=1600000)
parser.add_argument("--model_name", type=str, default="resnet")
parser.add_argument("--block_num", type=int, default=20)
parser.add_argument("--channel_num", type=int, default=512)
args = parser.parse_args()

# calc interval and warmup step
validation_interval = max(args.max_step // 40, 5)
print_interval = max(validation_interval // 500, 1)
warmup_step = min(args.max_step // 20, 80000)

with open("random_move_learn_settings.txt", "w") as f:
    f.write(f"learn_rate\t{args.learning_rate}\n")
    f.write(f"weight_decay\t{args.weight_decay}\n")
    f.write(f"batch_size\t{args.batch_size}\n")
    f.write(f"max_step\t{args.max_step}\n")
    f.write(f"learn_rate_decay_mode\t{args.learn_rate_decay_mode}\n")
    f.write(f"learn_rate_decay_period\t{args.learn_rate_decay_period}\n")
    f.write(f"warm_up_step\t{warmup_step}\n")
    f.write(f"model_name\t{args.model_name}\n")

block_num = args.block_num
channel_num = args.channel_num

model_class = model_dict[args.model_name]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# name
model_name_prefix = f"{args.model_name}_bl{block_num}_ch{channel_num}"

# prepare model
model = model_class(INPUT_CHANNEL_NUM, block_num, channel_num, POLICY_CHANNEL_NUM, BOARD_SIZE)
model.to(device)

# prepare predict_head
predict_head = torch.nn.Conv2d(in_channels=channel_num * 2, out_channels=POLICY_CHANNEL_NUM, kernel_size=1, padding=0)
predict_head.to(device)

# optimizer
optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
scheduler = LinearWarmupAndCooldownScheduler(optim, warmup_step, args.max_step)

# prepare output file
train_log = open("random_move_train_log.txt", "a")
header_text = "time	step\tpolicy_loss\taccuracy\tlearn_rate\n"
print(header_text, end="")
# 最初の学習のときだけヘッダーをファイルに書き込み
if scheduler.last_epoch == 0:
    train_log.write(header_text)

# timer start
start_time = time.time()

# train loop
continue_flag = True
while continue_flag:
    trainset = RandomMoveDataSet()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for batch in trainloader:
        # 指定ステップ数を超えていたら終了
        if scheduler.last_epoch >= args.max_step:
            continue_flag = False
            break

        x1, x2, policy_label = batch
        x1, x2, policy_label = x1.to(device), x2.to(device), policy_label.to(device)
        rep1 = model.forward_representation(x1)
        rep2 = model.forward_representation(x2)
        cat = torch.cat([rep1, rep2], dim=1)
        out = predict_head(cat)
        out = out.flatten(1)
        policy_loss = torch.nn.functional.cross_entropy(out, policy_label)
        pred = torch.argmax(out, 1)
        accuracy = (pred == policy_label).to(torch.float).mean()
        optim.zero_grad()
        policy_loss.backward()
        optim.step()
        scheduler.step()

        if scheduler.last_epoch % print_interval == 0:
            elapsed_sec = time.time() - start_time
            time_str = seconds_to_pretty_str(elapsed_sec)
            text = f"{time_str}\t{scheduler.last_epoch}\t{policy_loss.item():.4f}\t{accuracy.item() * 100:5.1f}\t{scheduler.get_last_lr()[0]:.5f}"
            print(text, end="\r")
            train_log.write(text + "\n")
            train_log.flush()

        if scheduler.last_epoch % validation_interval == 0:
            print()
            torch.save(model.state_dict(), f"{model_name_prefix}.pt")

torch.save(model.state_dict(), f"{model_name_prefix}.pt")
