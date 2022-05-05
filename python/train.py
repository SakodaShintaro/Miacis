import torch
from constant import *
from dataset import HcpeDataSet
import argparse
from common import seconds_to_pretty_str
from path_manager import PathManager
from scheduler import LinearWarmupAndCooldownScheduler
import time
from model_dict import model_dict

parser = argparse.ArgumentParser()
parser.add_argument("train_data_dir", type=str)
parser.add_argument("valid_data_dir", type=str)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--weight_decay", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--max_step", type=int, default=1600000)
parser.add_argument("--save_interval", type=int, default=1600001)
parser.add_argument("--learn_rate_decay_mode", type=int, default=0)
parser.add_argument("--learn_rate_decay_period", type=int, default=1600000)
parser.add_argument("--model_name", type=str, default="resnet")
parser.add_argument("--break_near_24h", action="store_true")
args = parser.parse_args()

validation_interval = max(args.max_step // 40, 5)
print_interval = max(validation_interval // 1000, 1)
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

block_num = 20
channel_num = 512

model_class = model_dict[args.model_name]

model = model_class(INPUT_CHANNEL_NUM, block_num, channel_num, POLICY_CHANNEL_NUM, BOARD_SIZE)
torch.save(model.state_dict(), f"{args.model_name}_bl{block_num}_ch{channel_num}_before_learn.pt")

path_manager = PathManager(args.train_data_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# optimizer
optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
scheduler = LinearWarmupAndCooldownScheduler(optim, warmup_step, args.max_step)

# valid data ファイルは水匠棋譜を決め打ち
validset = HcpeDataSet(f"{args.valid_data_dir}/suisho3kai-001.hcpe", is_valid=True)
validloader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

# prepare output file
train_log = open("supervised_train_log.txt", "w")
valid_log = open("supervised_valid_log.txt", "w")
header_text = "time	step\tpolicy_loss\tvalue_loss\tlearn_rate\n"
print(header_text, end="")
train_log.write(header_text)
valid_log.write(header_text)

def calc_loss(batch):
    x, policy_label, value_label = batch
    x, policy_label, value_label = x.to(device), policy_label.to(device), value_label.to(device)
    policy, value = model(x)
    policy = policy.flatten(1)
    policy_loss = torch.nn.functional.cross_entropy(policy, policy_label)
    value_loss = torch.nn.functional.cross_entropy(value, value_label)
    return policy_loss, value_loss

# timer start
start_time = time.time()

# train loop
total_step = 0
while total_step < args.max_step:
    curr_data_path = path_manager.get_next_path()
    trainset = HcpeDataSet(curr_data_path, is_valid=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for batch in trainloader:
        if total_step >= args.max_step:
            break
        total_step += 1

        model.train()
        policy_loss, value_loss = calc_loss(batch)
        optim.zero_grad()
        (policy_loss + value_loss).backward()
        optim.step()
        scheduler.step()

        if total_step % print_interval == 0:
            elapsed_sec = time.time() - start_time
            time_str = seconds_to_pretty_str(elapsed_sec)
            text = f"{time_str}\t{total_step}\t{policy_loss.item():.4f}\t{value_loss.item():.4f}\t{scheduler.get_last_lr()[0]:.5f}"
            print(text, end="\r")
            train_log.write(text + "\n")
            train_log.flush()

        if total_step % validation_interval == 0:
            model.eval()
            policy_loss_sum = 0
            value_loss_sum = 0
            for batch in validloader:
                with torch.no_grad():
                    policy_loss, value_loss = calc_loss(batch)
                    policy_loss_sum += policy_loss.item() * len(batch[0])
                    value_loss_sum += value_loss.item() * len(batch[0])
            policy_loss = policy_loss_sum / len(validset)
            value_loss = value_loss_sum / len(validset)
            elapsed_sec = time.time() - start_time
            time_str = seconds_to_pretty_str(elapsed_sec)
            text = f"{time_str}\t{total_step}\t{policy_loss:.4f}\t{value_loss:.4f}\t{scheduler.get_last_lr()[0]:.5f}\n"
            print(text, end="")
            valid_log.write(text)
            valid_log.flush()
            torch.save(model.state_dict(), f"{args.model_name}_bl{block_num}_ch{channel_num}.pt")

torch.save(model.state_dict(), f"{args.model_name}_bl{block_num}_ch{channel_num}.pt")
