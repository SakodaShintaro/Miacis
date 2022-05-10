#!/usr/bin/env python3
import torch
from constant import *
from dataset import HcpeDataSet
import argparse
from common import seconds_to_pretty_str
from path_manager import PathManager
from scheduler import LinearWarmupAndCooldownScheduler
import time
from model_dict import model_dict
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--weight_decay", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--max_step", type=int, default=1600000)
parser.add_argument("--save_interval", type=int, default=1600001)
parser.add_argument("--learn_rate_decay_mode", type=int, default=0)
parser.add_argument("--learn_rate_decay_period", type=int, default=1600000)
parser.add_argument("--model_name", type=str, default="resnet")
parser.add_argument("--block_num", type=int, default=20)
parser.add_argument("--channel_num", type=int, default=512)
parser.add_argument("--break_near_24h", action="store_true")
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

# calc interval and warmup step
validation_interval = max(args.max_step // 40, 5)
print_interval = max(validation_interval // 500, 1)
warmup_step = min(args.max_step // 20, 80000)

with open("supervised_learn_settings.txt", "w") as f:
    f.write(f"learn_rate\t{args.learning_rate}\n")
    f.write(f"weight_decay\t{args.weight_decay}\n")
    f.write(f"batch_size\t{args.batch_size}\n")
    f.write(f"max_step\t{args.max_step}\n")
    f.write(f"save_interval\t{args.save_interval}\n")
    f.write(f"learn_rate_decay_mode\t{args.learn_rate_decay_mode}\n")
    f.write(f"learn_rate_decay_period\t{args.learn_rate_decay_period}\n")
    f.write(f"warm_up_step\t{warmup_step}\n")
    f.write(f"model_name\t{args.model_name}\n")

block_num = args.block_num
channel_num = args.channel_num

model_class = model_dict[args.model_name]

# name
model_name_prefix = f"{args.model_name}_bl{block_num}_ch{channel_num}"
optimizer_name = "optimizer.pt"
scheduler_name = "scheduler.pt"

model = model_class(INPUT_CHANNEL_NUM, block_num, channel_num, POLICY_CHANNEL_NUM, BOARD_SIZE)

path_manager = PathManager(args.data_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# optimizer
optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
scheduler = LinearWarmupAndCooldownScheduler(optim, warmup_step, args.max_step)

# resume
if args.resume:
    # load model
    if os.path.exists(f"{model_name_prefix}.pt"):
        print(f"load {model_name_prefix}.pt")
        model.load_state_dict(torch.load(f"{model_name_prefix}.pt"))

    # load optimizer
    if os.path.exists(optimizer_name):
        print(f"load {optimizer_name}")
        optim.load_state_dict(torch.load(optimizer_name))

    # load scheduler
    if os.path.exists(scheduler_name):
        print(f"load {scheduler_name}")
        scheduler.load_state_dict(torch.load(scheduler_name))

# valid data ファイルはfloodgateの棋譜を決め打ち
validset = HcpeDataSet(f"{args.data_dir}/floodgate_2019-2021_r3500-001.hcpe", is_valid=True)
validloader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

# prepare output file
train_log = open("supervised_train_log.txt", "a")
valid_log = open("supervised_valid_log.txt", "a")
header_text = "time	step\tpolicy_loss\tvalue_loss\tlearn_rate\n"
print(header_text, end="")
# 最初の学習のときだけヘッダーをファイルに書き込み
if scheduler.last_epoch == 0:
    train_log.write(header_text)
    valid_log.write(header_text)

def calc_loss(batch, is_valid):
    x, policy_label, value_label = batch
    x, policy_label, value_label = x.to(device), policy_label.to(device), value_label.to(device)
    policy, value = model(x)
    policy = policy.flatten(1)
    policy_loss = torch.nn.functional.cross_entropy(policy, policy_label)

    if is_valid:
        value = torch.nn.functional.softmax(value, dim=1)
        value_item = [MIN_SCORE + VALUE_WIDTH * (i + 0.5) for i in range(BIN_SIZE)]
        value_item = torch.tensor(value_item, device=value.device)
        value = (value * value_item).sum(1)
        value_loss = torch.nn.functional.mse_loss(value, value_label)
    else:
        value_loss = torch.nn.functional.cross_entropy(value, value_label)
    return policy_loss, value_loss

# timer start
start_time = time.time()

# train loop
continue_flag = True
while continue_flag:
    curr_data_path = path_manager.get_next_path()
    trainset = HcpeDataSet(curr_data_path, is_valid=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for batch in trainloader:
        # Google Colaboratoryで動かすことを想定して24h近くになったら止める機能を準備する
        elapsed_sec = time.time() - start_time
        if args.break_near_24h and elapsed_sec >= 23.5 * 3600:
            continue_flag = False
            break

        # 指定ステップ数を超えていたら終了
        if scheduler.last_epoch >= args.max_step:
            continue_flag = False
            break

        model.train()
        policy_loss, value_loss = calc_loss(batch, is_valid=False)
        optim.zero_grad()
        (policy_loss + value_loss).backward()
        optim.step()
        scheduler.step()

        if scheduler.last_epoch % print_interval == 0:
            elapsed_sec = time.time() - start_time
            time_str = seconds_to_pretty_str(elapsed_sec)
            text = f"{time_str}\t{scheduler.last_epoch}\t{policy_loss.item():.4f}\t{value_loss.item():.4f}\t{scheduler.get_last_lr()[0]:.5f}"
            print(text, end="\r")
            train_log.write(text + "\n")
            train_log.flush()

        if scheduler.last_epoch % validation_interval == 0:
            model.eval()
            policy_loss_sum = 0
            value_loss_sum = 0
            for batch in validloader:
                with torch.no_grad():
                    policy_loss, value_loss = calc_loss(batch, is_valid=True)
                    policy_loss_sum += policy_loss.item() * len(batch[0])
                    value_loss_sum += value_loss.item() * len(batch[0])
            policy_loss = policy_loss_sum / len(validset)
            value_loss = value_loss_sum / len(validset)
            elapsed_sec = time.time() - start_time
            time_str = seconds_to_pretty_str(elapsed_sec)
            text = f"{time_str}\t{scheduler.last_epoch}\t{policy_loss:.4f}\t{value_loss:.4f}\t{scheduler.get_last_lr()[0]:.5f}\n"
            print(text, end="")
            valid_log.write(text)
            valid_log.flush()
            torch.save(model.state_dict(), f"{model_name_prefix}.pt")
            torch.save(optim.state_dict(), optimizer_name)
            torch.save(scheduler.state_dict(), scheduler_name)

torch.save(model.state_dict(), f"{model_name_prefix}.pt")
torch.save(optim.state_dict(), optimizer_name)
torch.save(scheduler.state_dict(), scheduler_name)
