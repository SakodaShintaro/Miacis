#!/usr/bin/env python3
import os
from generate_cnn_model import *
from generate_transformer_model import *
from generate_mlp_mixer_model import *

parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=str)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--no_message", action="store_true")
args = parser.parse_args()

input_channel_num = 42
board_size = 9
policy_channel_num = 27
input_tensor = torch.randn([args.batch_size, input_channel_num, board_size, board_size]).cuda()

script_model = torch.jit.load(args.model_path)

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

model = None
if "transformer" in args.model_path:
    model = TransformerModel(input_channel_num, block_num=block_num, channel_num=channel_num,
                             policy_channel_num=policy_channel_num,
                             board_size=board_size)
elif "mlp_mixer" in args.model_path:
    model = MLPMixer(input_channel_num, block_num=block_num, channel_num=channel_num,
                     policy_channel_num=policy_channel_num,
                     board_size=board_size)
else:
    model = CategoricalNetwork(input_channel_num, block_num=block_num, channel_num=channel_num,
                               policy_channel_num=policy_channel_num,
                               board_size=board_size)


model.load_state_dict(script_model.state_dict())

model.eval()
model.cuda()

save_path = args.model_path.replace(".model", ".onnx")
torch.onnx.export(model, input_tensor, save_path, dynamic_axes={"input": {0: "batch_size"}}, input_names=["input"])
if not args.no_message:
    print(f"export to {save_path}")
