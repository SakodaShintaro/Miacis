from model.resnet import CategoricalNetwork
import torch
from constant import *
from dataset import MiacisDataSet
import argparse
from scheduler import LinearWarmupAndCooldownScheduler

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--weight_decay", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--max_step", type=int, default=1600000)
parser.add_argument("--validation_interval", type=int, default=40000)
parser.add_argument("--save_interval", type=int, default=1600001)
parser.add_argument("--learn_rate_decay_mode", type=int, default=0)
parser.add_argument("--learn_rate_decay_period", type=int, default=1600000)
parser.add_argument("--clip_grad_norm", type=float, default=10)
parser.add_argument("--model_name", type=str, default=None)
parser.add_argument("--break_near_24h", action="store_true")
args = parser.parse_args()

warmup_step = min(args.max_step // 20, 80000)

with open("supervised_learn_settings.txt", "w") as f:
    f.write(f"learn_rate\t{args.learning_rate}\n")
    f.write(f"weight_decay\t{args.weight_decay}\n")
    f.write(f"batch_size\t{args.batch_size}\n")
    f.write(f"max_step\t{args.max_step}\n")
    f.write(f"validation_interval\t{args.validation_interval}\n")
    f.write(f"save_interval\t{args.save_interval}\n")
    f.write(f"learn_rate_decay_mode\t{args.learn_rate_decay_mode}\n")
    f.write(f"learn_rate_decay_period\t{args.learn_rate_decay_period}\n")
    f.write(f"warm_up_step\t{warmup_step}\n")
    f.write(f"clip_grad_norm\t{args.clip_grad_norm}\n")
    f.write(f"model_name\t{args.model_name}\n")

block_num = 20
channel_num = 512
policy_channel_num = 27

model = CategoricalNetwork(INPUT_CHANNEL_NUM, block_num, channel_num, policy_channel_num, BOARD_SIZE)

trainset = MiacisDataSet('/home/sakoda/data/ShogiAIBookData/dlshogi_with_gct-001.hcpe')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# optimizer
optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
scheduler = LinearWarmupAndCooldownScheduler(optim, warmup_step, args.max_step)

for step, batch in enumerate(trainloader):
    if step >= args.max_step:
        break
    x, policy_label, value_label = batch
    x, policy_label, value_label = x.to(device), policy_label.to(device), value_label.to(device)

    policy, value = model(x)

    policy = policy.flatten(1)

    policy_loss = torch.nn.functional.cross_entropy(policy, policy_label)

    value_loss = torch.nn.functional.cross_entropy(value, value_label)

    text = f"{step}\t{policy_loss.item():.4f}\t{value_loss.item():.4f}\t{scheduler.get_last_lr()[0]:.5f}\n"

    print(text, end="")
    optim.zero_grad()
    (policy_loss + value_loss).backward()
    optim.step()
    scheduler.step()


torch.save(model.state_dict(), "model.pt")
