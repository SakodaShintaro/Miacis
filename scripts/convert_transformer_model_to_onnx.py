#!/usr/bin/env python3
from generate_transformer_model import *

parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=str)
parser.add_argument("--block_num", type=int, default=10)
parser.add_argument("--channel_num", type=int, default=256)
args = parser.parse_args()

batch_size = 4
input_channel_num = 42
board_size = 9
policy_channel_num = 27
input_tensor = torch.randn([batch_size, input_channel_num, board_size, board_size]).cuda()

script_model = torch.jit.load(args.model_path)

model = TransformerModel(input_channel_num, block_num=args.block_num, channel_num=args.channel_num,
                         policy_channel_num=policy_channel_num,
                         board_size=board_size)
model.load_state_dict(script_model.state_dict())

model.eval()
model.cuda()

save_path = args.model_path.replace(".model", ".onnx")
torch.onnx.export(model, input_tensor, save_path)
print(f"export to {save_path}")
