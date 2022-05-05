from model.resnet import CategoricalNetwork
import torch
from constant import *
from dataset import MiacisDataSet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--saved_model_path", type=str, default=None)
parser.add_argument("--learning_rate", type=float, default=0.01)
args = parser.parse_args()

block_num = 20
channel_num = 512
policy_channel_num = 27

model = CategoricalNetwork(INPUT_CHANNEL_NUM, block_num, channel_num, policy_channel_num, BOARD_SIZE)

trainset = MiacisDataSet('/home/sakoda/data/ShogiAIBookData/dlshogi_with_gct-001.hcpe')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# optimizer
optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epoch)

for step, batch in enumerate(trainloader):
    x, policy_label, value_label = batch
    x, policy_label, value_label = x.to(device), policy_label.to(device), value_label.to(device)

    policy, value = model(x)

    policy = policy.flatten(1)

    policy_loss = torch.nn.functional.cross_entropy(policy, policy_label)

    value_loss = torch.nn.functional.cross_entropy(value, value_label)

    print(step, policy_loss.item(), value_loss.item())
    optim.zero_grad()
    (policy_loss + value_loss).backward()
    optim.step()

torch.save(model.state_dict(), "model.pt")
